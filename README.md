# FraudCopilot — Agentic Fraud Investigation System

An analyst-in-the-loop fraud investigation application that combines **deterministic fraud rules, a scikit-learn risk model, retrieval of similar historical cases, and an LLM-generated structured report**. A human analyst reviews the evidence and records the final decision.

> **Portfolio / educational project:** the repository uses synthetic transaction and case data. It is not connected to a bank, payment network, or production fraud system.

## What it does

Given a transaction ID, FraudCopilot builds an investigation context and runs a five-stage LangGraph workflow. The Streamlit interface presents the transaction, evidence, risk assessment, recommended action, and analyst review controls.

### Key capabilities

- **Transaction context:** transaction details, device/IP metadata, and recent customer history.
- **Explainable rules:** checks for high-risk merchants, unusual locations, odd-hour activity, new devices, and unusually high amounts versus customer history.
- **Similar-case retrieval:** TF-IDF vectors and cosine similarity retrieve up to three related historical case notes.
- **ML risk scoring:** a Random Forest pipeline returns a fraud probability and a high-/low-risk label.
- **Structured LLM report:** Groq generates a Pydantic-validated investigation report with verdict, confidence, summary, evidence, and recommended action.
- **Human review:** an analyst can approve or override the suggested verdict and optionally add a note. Decisions are appended to a local CSV log.

## Architecture

```text
                    Selected transaction ID
                              |
                              v
                    1. Gather context
                 transaction + device + history
                              |
                              v
                    2. Run rule checks
                    deterministic signals
                              |
                              v
                    3. Retrieve similar cases
                    TF-IDF + cosine similarity
                              |
                              v
                    4. Score risk
                    Random Forest pipeline
                              |
                              v
                    5. Write investigation report
                    Groq LLM + Pydantic schema
                              |
                              v
                    Human analyst review
                       approve / override
                              |
                              v
                    Local decision log (CSV)
```

LangGraph orchestrates the workflow as a sequential state graph. The LLM is used for report drafting after the context, rules, retrieval, and model-scoring steps have produced their outputs.

## Technology

| Area | Tools |
|---|---|
| User interface | Streamlit |
| Agent orchestration | LangGraph |
| LLM integration | LangChain, ChatGroq |
| Structured output | Pydantic |
| Machine learning | scikit-learn, Random Forest |
| Case retrieval | TF-IDF, cosine similarity |
| Data processing | Python, pandas, NumPy |
| Visualisation | Plotly |

## Repository structure

```text
FraudCopilot/
├── app.py
├── generate_synthetic_data.py
├── requirements.txt
├── runtime.txt
├── .env.example
├── data/
│   ├── transactions.csv
│   ├── device_ip_logs.csv
│   ├── past_cases.csv
│   └── labeled_dataset.csv
├── models/
│   └── risk_model.pkl
└── src/
    ├── report_schema.py
    ├── train_risk_model.py
    ├── agents/
    │   └── graph.py
    └── tools/
        ├── context_tools.py
        ├── retrieval.py
        ├── risk_model_tool.py
        └── rule_engine.py
```

## Getting started

### Requirements

- Python 3.10 or a compatible Python version
- A Groq API key for generating investigation reports
- Git

### Clone and install

```bash
git clone https://github.com/Mimshad1999/FraudCopilot.git
cd FraudCopilot

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure the API key

Copy the example environment file:

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a key from the [Groq Console](https://console.groq.com/keys). Never commit your real `.env` file or expose API keys in screenshots.

## Run the application

The repository includes the synthetic CSV datasets and a trained model file. From the project root, run:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (typically [http://localhost:8501](http://localhost:8501)).

1. Select a transaction from the sidebar.
2. Click **Run Investigation**.
3. Review the generated verdict, confidence, evidence, reasoning, and recommended action.
4. Choose **Approve** or **Override** and optionally add a note.
5. Open **Your Review History** to inspect decisions recorded in the local log.

The LLM report step requires a valid `GROQ_API_KEY`.

## Train the risk model

To retrain the Random Forest model using the committed labeled dataset:

```bash
python src/train_risk_model.py
```

The script uses a stratified 80/20 train/test split and prints precision, recall, F1-score, a confusion matrix, and a classification report. It saves the fitted pipeline to `models/risk_model.pkl`.

To regenerate the synthetic datasets:

```bash
python generate_synthetic_data.py
```

The generator creates synthetic transaction, device/IP, past-case, and labeled data. Regenerate data before retraining only if you intentionally want a new synthetic sample.

## Run an investigation from the command line

After dependencies, data, model, and API key are configured:

```bash
python -m src.agents.graph TXN100002
```

Replace `TXN100002` with a transaction ID present in `data/transactions.csv`.

## Important notes and limitations

- **Synthetic data only:** the included generator creates 2,000 synthetic transactions and simulated device/IP and case-history records. No real customer or bank data is used.
- **Not production validated:** the synthetic patterns are deliberately generated and may be easier for a model to separate than real-world fraud. Metrics from this dataset must not be presented as expected real-world performance.
- **Human decision remains essential:** the generated report is decision support, not an automated final fraud determination.
- **Retrieval method:** the current implementation uses TF-IDF and cosine similarity, not a vector database or embedding-based retriever.
- **LLM provider:** the current graph uses Groq via `ChatGroq`. The model identifier is configured in `src/agents/graph.py`.
- **Local review log:** analyst decisions are written to `data/analyst_decisions.csv` on the machine running the app.

## Potential next improvements

- Add tests for rule checks, retrieval, model scoring, and graph execution.
- Add model monitoring and evaluation on a more realistic, independently sourced dataset.
- Compare TF-IDF retrieval with an embedding-based retriever.
- Add investigation latency and token/cost logging.
- Add authentication and persistent storage before any multi-user deployment.

## Author

**Mimshad Alam**  
[GitHub](https://github.com/Mimshad1999) · [LinkedIn](https://www.linkedin.com/in/mimshad-alam-995a20230/)
