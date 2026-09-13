"""
Agentic Fraud Investigation Copilot — Streamlit UI (Analyst Edition)
=====================================================================
Run:
    streamlit run app.py

Designed for a fraud analyst reviewing agent-generated reports:
- Dashboard summary stats at the top
- Sidebar transaction picker with search/filter
- Color-coded verdict badges + visual risk gauge
- Evidence broken out by source, in plain language
- One-click Approve / Override with a note field
- Decision log analysts can scroll back through
"""

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
LOG_PATH = DATA_DIR / "analyst_decisions.csv"

st.set_page_config(
    page_title="Fraud Investigation Copilot",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }

    .verdict-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 6px;
    }
    .verdict-fraud        { background:#fde2e2; color:#b3261e; }
    .verdict-likely       { background:#ffe9d6; color:#b56a00; }
    .verdict-review       { background:#fff6d6; color:#8a6d00; }
    .verdict-legit        { background:#dcf3e3; color:#1e7e34; }

    .evidence-card {
        background:#f7f8fa;
        border-left: 4px solid #4a6fa5;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.93rem;
    }
    .evidence-source {
        font-weight: 700;
        color: #4a6fa5;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
    }

    .metric-card {
        background: white;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 14px 18px;
        text-align:center;
    }

    div[data-testid="stExpander"] { border: none; }
</style>
""", unsafe_allow_html=True)

VERDICT_STYLE = {
    "Fraud":                {"class": "verdict-fraud",  "emoji": "🔴"},
    "Likely Fraud":         {"class": "verdict-likely",  "emoji": "🟠"},
    "Needs Manual Review":  {"class": "verdict-review",  "emoji": "🟡"},
    "Legitimate":           {"class": "verdict-legit",   "emoji": "🟢"},
}


def risk_gauge(confidence: float, verdict: str):
    color = {
        "Fraud": "#b3261e", "Likely Fraud": "#b56a00",
        "Needs Manual Review": "#8a6d00", "Legitimate": "#1e7e34",
    }.get(verdict, "#4a6fa5")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 30}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#eef0f4"},
                {"range": [40, 70], "color": "#e4e8f0"},
                {"range": [70, 100], "color": "#dbe1ec"},
            ],
        },
        title={"text": "Agent Confidence", "font": {"size": 14}},
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
transactions = pd.read_csv(DATA_DIR / "transactions.csv")
transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
transactions = transactions.sort_values("timestamp", ascending=False)

if "report" not in st.session_state:
    st.session_state["report"] = None
if "report_txn_id" not in st.session_state:
    st.session_state["report_txn_id"] = None

# ---------------------------------------------------------------------
# Header + dashboard stats
# ---------------------------------------------------------------------
st.title("🔎 Fraud Investigation Copilot")
st.caption("Agent-drafted investigation reports for analyst review — you always have the final call.")

if not os.environ.get("GROQ_API_KEY"):
    st.warning("No `GROQ_API_KEY` found. Add it to your `.env` file to run investigations.", icon="⚠️")

stat_cols = st.columns(4)
with stat_cols[0]:
    st.markdown(f'<div class="metric-card"><h3>{len(transactions)}</h3>Total Transactions</div>', unsafe_allow_html=True)
with stat_cols[1]:
    high_amt = (transactions["amount_inr"] > transactions["amount_inr"].quantile(0.9)).sum()
    st.markdown(f'<div class="metric-card"><h3>{high_amt}</h3>Top 10% by Amount</div>', unsafe_allow_html=True)
with stat_cols[2]:
    n_customers = transactions["customer_id"].nunique()
    st.markdown(f'<div class="metric-card"><h3>{n_customers}</h3>Unique Customers</div>', unsafe_allow_html=True)
with stat_cols[3]:
    n_reviewed = pd.read_csv(LOG_PATH).shape[0] if LOG_PATH.exists() else 0
    st.markdown(f'<div class="metric-card"><h3>{n_reviewed}</h3>Reviewed by You</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------
# Sidebar — transaction picker
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Transaction Queue")
    search = st.text_input("Search by Transaction or Customer ID", "")
    sort_by_amount = st.checkbox("Sort by highest amount first", value=False)

    filtered = transactions
    if search:
        filtered = filtered[
            filtered["transaction_id"].str.contains(search, case=False)
            | filtered["customer_id"].str.contains(search, case=False)
        ]
    if sort_by_amount:
        filtered = filtered.sort_values("amount_inr", ascending=False)

    st.caption(f"{len(filtered)} transactions match")

    txn_id = st.radio(
        "Select a transaction",
        options=filtered["transaction_id"].head(30).tolist(),
        format_func=lambda tid: f"{tid} — ₹{transactions.loc[transactions.transaction_id==tid,'amount_inr'].values[0]:,.0f}",
    ) if len(filtered) > 0 else None

# ---------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------
if txn_id:
    selected = transactions[transactions["transaction_id"] == txn_id].iloc[0]

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.subheader("Transaction Details")
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | **Transaction ID** | `{selected['transaction_id']}` |
        | **Customer** | {selected['customer_name']} ({selected['customer_id']}) |
        | **Amount** | ₹{selected['amount_inr']:,.2f} |
        | **Merchant** | {selected['merchant']} |
        | **City** | {selected['city']} |
        | **Time** | {selected['timestamp']} |
        """)

        run_button = st.button("🚀 Run Investigation", type="primary", use_container_width=True)

        if run_button:
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Set GROQ_API_KEY in your .env file first.")
            else:
                with st.spinner("Gathering context → checking rules → retrieving similar cases → scoring risk → drafting report..."):
                    from src.agents.graph import investigate
                    st.session_state["report"] = investigate(txn_id)
                    st.session_state["report_txn_id"] = txn_id

    with right:
        st.subheader("Investigation Report")
        report = st.session_state["report"]

        if report and st.session_state["report_txn_id"] == txn_id:
            style = VERDICT_STYLE.get(report.verdict, {"class": "verdict-review", "emoji": "⚪"})

            badge_col, gauge_col = st.columns([1.3, 1])
            with badge_col:
                st.markdown(
                    f'<div class="verdict-badge {style["class"]}">{style["emoji"]} {report.verdict}</div>',
                    unsafe_allow_html=True,
                )
                st.write(report.summary)
            with gauge_col:
                st.plotly_chart(risk_gauge(report.confidence, report.verdict), use_container_width=True)

            st.markdown("#### 🧾 Evidence")
            for item in report.evidence:
                st.markdown(f"""
                <div class="evidence-card">
                    <div class="evidence-source">{item.source}</div>
                    {item.finding}
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🧠 Agent Reasoning (why this verdict?)"):
                st.write(report.reasoning)

            st.markdown("#### ✅ Recommended Action")
            st.info(report.recommended_action)

            st.divider()
            st.markdown("#### Your Decision")
            d1, d2, d3 = st.columns([1, 1, 2])
            with d1:
                approve = st.button("✅ Approve", use_container_width=True)
            with d2:
                reject = st.button("❌ Override", use_container_width=True)
            with d3:
                note = st.text_input("Note (optional)", label_visibility="collapsed", placeholder="Add a note (optional)")

            if approve or reject:
                decision = "approved" if approve else "rejected"
                log_row = pd.DataFrame([{
                    "transaction_id": report.transaction_id,
                    "agent_verdict": report.verdict,
                    "agent_confidence": report.confidence,
                    "analyst_decision": decision,
                    "analyst_note": note,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }])
                log_row.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
                st.success(f"Decision logged: **{decision}** for {report.transaction_id}")
        else:
            st.info("Click **Run Investigation** to generate a report for this transaction.")

else:
    st.info("No transactions match your search. Try clearing the sidebar filter.")

# ---------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------
st.divider()
with st.expander("📜 Your Review History"):
    if LOG_PATH.exists():
        log_df = pd.read_csv(LOG_PATH).sort_values("timestamp", ascending=False)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No decisions logged yet — approve or override a report to start your history.")