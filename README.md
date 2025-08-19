 Agentic Preprocessor Demo (GPT-Powered)

This project is a beginner-friendly Agentic AI demo that uses an LLM agent (OpenAI GPT) to inspect a dataset schema and autonomously decide preprocessing steps.
It combines:

- Automatic schema analysis

- Smart decision-making using GPT as an agent

- Handling missing values, outliers, and duplicates

- Data type conversion + categorical encoding

- Transparent reporting of all actions

 Key Concepts

Agentic AI: GPT acts as a data preprocessing agent, deciding what steps are needed rather than applying all rules blindly.

Adaptive Cleaning: Different datasets → different preprocessing strategies.

Explainability: A report.txt is generated explaining exactly what was done.

Demonstration ready: Ideal for LinkedIn, Medium, or interview portfolios.

 How It Works

Load dataset (CSV file).

Summarize schema (dtypes, null counts, duplicates, categorical candidates).

Ask GPT to recommend preprocessing steps.

Apply steps (imputation, deduplication, outlier handling, encoding).

Save:

Cleaned dataset (output/retailcustomerfeedbackdataset_cleaned.csv)

Report (output/report.txt) with schema + decisions + actions.

 Quick Start
1. Clone the repository
git clone https://github.com/JohnDBSDSN/agentic-preprocessor-demo.git
cd agentic-preprocessor-demo

2. Add your OpenAI API key

Create a .env file in the root folder:

OPENAI_API_KEY=sk-...


Never share this key — it’s already excluded via .gitignore.

3. Create virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate   # Windows
# or source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

4. Run the preprocessor
cd src
python agentic_preprocessor.py

Project Structure
agentic-preprocessor-demo/
│
├── src/
│   ├── agentic_preprocessor.py   # Main agent logic
│   ├── utils.py                  # Helper functions (schema, loader, printer)
│
├── data/
│   └── retailcustomerfeedbackdataset.csv   # Input dataset
│
├── output/
│   ├── retailcustomerfeedbackdataset_cleaned.csv   # Cleaned data
│   └── report.txt                                # Preprocessing report
│
├── .env                  # Your OpenAI API key (excluded from Git)
├── .gitignore
├── requirements.txt
└── README.md

 Sample Output

Example report.txt (excerpt):

=== AGENTIC PREPROCESSOR REPORT ===

Agent Decision Report (GPT)
{
  "selected_steps": ["missing_data", "dtype_conversion", "deduplicate", "outliers"],
  "explanation": "The dataset has missing values, duplicates, and potential outliers in numeric columns. Dates need conversion."
}

Preprocessing Steps Applied
- missing_data: Imputed numerics with median, categoricals with mode.
- dtype_conversion: Converted dates, cast low-cardinality objects.
- deduplicate: Dropped 3 duplicate rows.
- outliers: Applied IQR capping on numeric columns.

Final Dataset Shape
100 rows x 12 cols

 Why This Matters

Traditional preprocessing = apply all steps blindly.
Agentic preprocessing = GPT decides only the needed steps, and explains why.

This showcases reasoning, autonomy, decision-making, and transparency — the pillars of Agentic AI.

 Built With
    - Python
    - Pandas
    - Scikit-learn
    - OpenAI GPT
    - dotenv

 Disclaimer

This demo uses an OpenAI key stored locally in .env. Make sure to:

NEVER commit your .env

Use python-dotenv to keep it secure

 Made with  by John Daniel