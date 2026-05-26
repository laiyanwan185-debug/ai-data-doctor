# Data Doctor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Shapiro--Wilk-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![AI Workflow](https://img.shields.io/badge/AI-Code%20Generation-111827?style=for-the-badge&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/Project-Data%20%2B%20AI%20Exploration-10B981?style=for-the-badge)

**A statistical data-cleaning platform powered by Shapiro-Wilk diagnostics and AI-generated Pandas code**

**Data Doctor** is a practical Data + AI project focused on turning dataset diagnosis into executable cleaning actions.
It combines **statistical testing (Shapiro-Wilk)** with **LLM-driven code generation (Code Interpreter style)** to create a more transparent, auditable, and workflow-oriented data cleaning experience.

</div>

---

## Why This Project

Most data cleaning tools stop at showing charts, warnings, or summary statistics.
Data Doctor goes one step further:

- it **uploads real tabular data** from CSV or Excel,
- runs **statistical diagnosis** instead of only surface-level inspection,
- uses **skewness and kurtosis** to understand distribution shape,
- asks the model to **generate Pandas cleaning code** based on those signals,
- lets the user **review and safely execute** the generated code,
- and finally **exports the cleaned dataset** for downstream analysis or modeling.

This makes the project not just a dashboard, but a full **diagnose -> reason -> generate -> execute -> export** pipeline.

---

## Core Workflow

```text
Upload Dataset
    -> Statistical Diagnosis
    -> Shapiro-Wilk Normality Test
    -> Skewness / Kurtosis Analysis
    -> AI Generates Pandas Cleaning Code
    -> Safe Execution in Controlled Namespace
    -> Before/After Comparison
    -> Export Clean Data
```

### Workflow Breakdown

| Step | What happens | Why it matters |
|------|--------------|----------------|
| **1. Upload** | Import `.csv`, `.xlsx`, `.xls` files | Supports common real-world data entry points |
| **2. Statistical diagnosis** | Scan missing values, column types, descriptive statistics | Builds a structured understanding of dataset health |
| **3. Normality testing** | Run **Shapiro-Wilk** on numeric columns | Decides whether cleaning should follow normal vs non-normal assumptions |
| **4. Distribution analysis** | Measure **skewness** and **kurtosis** | Detects asymmetric and heavy-tail behavior |
| **5. AI code generation** | Generate **Pandas** cleaning code from diagnosis report | Turns diagnosis into reproducible actions |
| **6. Safe execution** | Execute code on a copied DataFrame in a controlled namespace | Keeps the workflow auditable and user-reviewed |
| **7. Export** | Download cleaned data as CSV | Makes the result directly usable in analysis pipelines |

---

## What Makes Data Doctor Different

### 1. Statistics-first, not prompt-first
Instead of asking a model to clean blindly, Data Doctor first computes structured evidence:

- missing-value profile
- descriptive statistics
- **Shapiro-Wilk normality test**
- **skewness**
- **kurtosis**
- type anomaly detection

This gives the model a stronger statistical basis for deciding **mean / median / mode imputation**, **3-sigma vs IQR outlier handling**, and **distribution transformation suggestions**.

### 2. AI-generated cleaning code you can inspect
The platform does not hide its actions in a black box.
It generates **readable Pandas code**, shows it to the user, and lets the user decide whether to execute it.

That means the cleaning process is:

- inspectable,
- reproducible,
- editable,
- and easier to trust.

### 3. Before/after evaluation is built in
After execution, the app re-runs diagnostics on the cleaned dataset and compares:

- missing values,
- number of normally distributed columns,
- average absolute skewness,
- per-column improvements.

The result is a cleaning loop with measurable feedback rather than one-shot automation.

---

## Feature Highlights

- **CSV / Excel upload** for fast dataset intake
- **Automatic numeric vs categorical column classification**
- **Missing value scan** with column-level visibility
- **Descriptive statistics dashboard**
- **Shapiro-Wilk normality testing** with performance-aware sampling
- **Skewness and kurtosis analysis** for shape diagnostics
- **Type anomaly detection** for suspicious column inference
- **Plotly-based distribution explorer**
- **AI-generated Pandas cleaning code** based on statistical diagnosis
- **Safe execution flow** with user review before running code
- **Before/after comparison metrics** for data quality improvement
- **CSV export** for cleaned output

---

## AI Cleaning Logic

The platform uses a statistics-oriented system prompt to guide code generation.
Its cleaning philosophy is roughly:

- **Missing values**
  - normal distribution -> prefer **mean** imputation
  - non-normal or strongly skewed distribution -> prefer **median** imputation
  - categorical variables -> prefer **mode** or explicit missing labels

- **Outliers**
  - normal distribution -> prefer **3-sigma** handling
  - non-normal distribution -> prefer **IQR-based capping**

- **Distribution optimization**
  - high absolute skewness -> consider **log transform** or **Box-Cox style ideas** where appropriate

This makes the generated code more aligned with statistical reasoning than naive rule-of-thumb cleaning.

---

## Project Architecture

```text
app.py
 ├─ File upload and data loading
 ├─ Dataset profiling and diagnosis
 │   ├─ missing value analysis
 │   ├─ Shapiro-Wilk test
 │   ├─ skewness / kurtosis computation
 │   └─ type anomaly detection
 ├─ Visualization layer
 ├─ AI code generation layer
 ├─ Controlled code execution layer
 ├─ Before/after comparison layer
 └─ Clean data export
```

---

## Tech Stack

| Layer | Tools |
|------|------|
| **Frontend / App** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Statistical Testing** | SciPy |
| **Visualization** | Plotly |
| **LLM Code Generation** | OpenAI-compatible client / DeepSeek API |
| **Output** | Executable Pandas cleaning code + cleaned CSV |

---

## Quick Start

### 1. Clone the project

```bash
git clone <your-repo-url>
cd ai-data-doctor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

You can provide your API key in either of these ways:

```bash
export DEEPSEEK_API_KEY="your_api_key"
```

Or enter it manually in the Streamlit sidebar when the app starts.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Recommended Usage Flow

1. Upload a CSV or Excel dataset.
2. Inspect missing values, descriptive statistics, and column types.
3. Review **Shapiro-Wilk**, **skewness**, and **kurtosis** results.
4. Let the model generate Pandas cleaning code.
5. Audit the generated code before execution.
6. Execute the cleaning step in the app.
7. Compare before/after diagnostics.
8. Export the cleaned dataset.

---

## Best Use Cases

Data Doctor is especially useful for:

- exploratory data analysis workflows,
- tabular dataset preprocessing,
- teaching or demonstrating statistics-aware cleaning,
- building trustable AI-assisted analytics tools,
- experimenting with **Data + AI** product design.

---

## Why It Matters in My Data + AI Exploration

This project represents a hands-on exploration of an important question:

> Can we combine rigorous statistical diagnosis with generative AI so that data cleaning becomes faster, smarter, and still auditable?

Data Doctor is my practical answer to that question.
It is not just about using an LLM for convenience; it is about designing a workflow where:

- **statistics provides the evidence**,
- **AI generates the action**,
- **the user keeps control**,
- and **the result stays reproducible**.

For anyone exploring the intersection of **data analysis, statistical reasoning, and AI-assisted automation**, this project is a strong real-world prototype.

---

## Current Strengths

- clear diagnosis-to-action workflow
- transparent AI-generated code review step
- practical support for real tabular data files
- measurable before/after quality comparison
- strong positioning as a **Data + AI** portfolio project

## Possible Next Iterations

- sandboxed execution for stronger runtime isolation
- richer column-level transformation suggestions
- multi-step cleaning plans instead of single-pass generation
- data quality scoring and report export
- model comparison across different code-generation backends

---

## License

This project is intended for learning, experimentation, and portfolio demonstration.
Add a formal license here if you plan to open-source it publicly.

---

## Final Note

If you are interested in the future of **AI-assisted analytics**, **statistical automation**, and **explainable data tooling**, Data Doctor is exactly the kind of project worth building.

It is a practical experiment in making data cleaning:

**more statistical, more transparent, and more executable.**
