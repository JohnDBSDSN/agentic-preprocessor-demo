"""
Agentic Preprocessor Demo - GPT Powered
---------------------------------------

This script uses GPT to act as an *Agent* for preprocessing decisions.
Instead of hard-coded rules, GPT inspects the dataset schema and decides
which preprocessing steps to apply.

Core Preprocessing Activities:
1. Handle missing data (numerics → median, categoricals → mode, dates → median date).
2. Data type conversion (parse date-like strings, cast low-cardinality objects as categorical).
3. Remove duplicate rows.
4. Detect and cap outliers (IQR method on numeric columns).
5. Encode categorical columns (One-Hot Encoding).

Workflow:
- Load dataset from `../data/retailcustomerfeedbackdataset.csv`.
- Summarize schema.
- Ask GPT to recommend preprocessing steps.
- Apply steps.
- Save cleaned dataset + report in `../output/`.

Usage:
    python src/agentic_preprocessor.py
"""

import os
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
from openai import OpenAI

from utils import load_csv, brief_schema, print_section

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Helper: cap outliers using IQR method ---
def iqr_cap(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower, upper)


# --- Ask GPT to decide preprocessing ---
def decide_with_gpt(schema: dict) -> dict:
    system_prompt = """You are a data preprocessing agent.
Given a dataset schema, decide which preprocessing steps to apply.
Valid steps: ["missing_data", "dtype_conversion", "deduplicate", "outliers", "categorical_encoding"].
Respond in strict JSON: { "selected_steps": [...], "explanation": "..." }"""

    user_prompt = f"Schema: {json.dumps(schema, indent=2, default=str)}"

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    try:
        decision = json.loads(raw)
    except Exception:
        decision = {"selected_steps": ["missing_data", "dtype_conversion"], "explanation": raw}
    return decision


# --- Apply preprocessing steps based on GPT decision ---
def apply_preprocessing(df: pd.DataFrame, steps: list, target: str = "label") -> tuple:
    report = {}
    work = df.copy()

    # 1. Data type conversion
    if "dtype_conversion" in steps:
        for c in work.columns:
            if work[c].dtype == "object":
                try:
                    converted = pd.to_datetime(work[c], errors="coerce", dayfirst=True)
                    if converted.notna().sum() > 0:
                        work[c] = converted
                except Exception:
                    pass

        # Impute missing dates with median
        for c in work.select_dtypes(include="datetime64[ns]").columns:
            if work[c].isna().sum() > 0:
                median_date = work[c].dropna().median()
                missing_count = work[c].isna().sum()
                work[c] = work[c].fillna(median_date)
                report[f"date_imputation_{c}"] = (
                    f"Resolved {missing_count} missing values in '{c}' using Median method → {median_date.date()}"
                )

        # Cast low-cardinality objects to category
        for c in work.select_dtypes(include="object").columns:
            if work[c].nunique() <= max(50, int(0.2 * len(work))):
                work[c] = work[c].astype("category")

        report["dtype_conversion"] = "Converted dates, cast low-cardinality objects."

    # 2. Duplicates
    if "deduplicate" in steps:
        before = len(work)
        work = work.drop_duplicates()
        report["deduplicate"] = f"Dropped {before - len(work)} duplicate rows."

    # 3. Missing data
    if "missing_data" in steps:
        num_cols = [c for c in work.columns if c != target and pd.api.types.is_numeric_dtype(work[c])]
        cat_cols = [c for c in work.columns if c != target and (work[c].dtype == "object" or str(work[c].dtype) == "category")]
        if num_cols:
            imputer_num = SimpleImputer(strategy="median")
            work[num_cols] = imputer_num.fit_transform(work[num_cols])
        if cat_cols:
            imputer_cat = SimpleImputer(strategy="most_frequent")
            work[cat_cols] = imputer_cat.fit_transform(work[cat_cols])
        report["missing_data"] = "Imputed numerics with median, categoricals with mode."

    # 4. Outliers
    if "outliers" in steps:
        for c in work.columns:
            if c != target and pd.api.types.is_numeric_dtype(work[c]):
                work[c] = iqr_cap(work[c])
        report["outliers"] = "Applied IQR capping on numeric columns."

    # 5. Categorical encoding
    if "categorical_encoding" in steps:
        categorical_cols = [c for c in work.columns if c != target and (str(work[c].dtype) == "category" or work[c].dtype == "object")]
        if categorical_cols:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ohe_df = pd.DataFrame(
                ohe.fit_transform(work[categorical_cols]),
                columns=ohe.get_feature_names_out(categorical_cols),
                index=work.index,
            )
            work = pd.concat([work.drop(columns=categorical_cols), ohe_df], axis=1)
        report["categorical_encoding"] = f"Applied One-Hot Encoding to: {categorical_cols}"

    return work, report


# --- Main driver ---
def main(input_csv: str = "../data/retailcustomerfeedbackdataset.csv", target: str = "label"):
    print_section("LOAD & INSPECT")
    df = load_csv(input_csv)
    schema = brief_schema(df)
    print(json.dumps(schema, indent=2, default=str))

    print_section("AGENT ANALYSIS (GPT)")
    decision = decide_with_gpt(schema)
    print("Selected steps:", decision["selected_steps"])
    print("Explanation:", decision["explanation"])

    print_section("APPLY PREPROCESSING")
    clean_df, step_report = apply_preprocessing(df, decision["selected_steps"], target=target)
    for k, v in step_report.items():
        print(f"- {k}: {v}")

    # Save outputs
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)

    cleaned_path = os.path.join(output_dir, "retailcustomerfeedbackdataset_cleaned.csv")
    clean_df.to_csv(cleaned_path, index=False)

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== AGENTIC PREPROCESSOR REPORT ===\n\n")
        f.write("Schema Summary\n")
        f.write(json.dumps(schema, indent=2, default=str) + "\n\n")
        f.write("Agent Decision Report (GPT)\n")
        f.write(json.dumps(decision, indent=2) + "\n\n")
        f.write("Preprocessing Steps Applied\n")
        for k, v in step_report.items():
            f.write(f"- {k}: {v}\n")
        f.write("\nFinal Dataset Shape\n")
        f.write(f"{clean_df.shape[0]} rows x {clean_df.shape[1]} cols\n")

    print_section("FINAL DATASET PREVIEW")
    print(clean_df.head())
    print(f"\n Cleaned dataset saved as: {cleaned_path}")
    print(f" Report saved as: {report_path}")


if __name__ == "__main__":
    main()
