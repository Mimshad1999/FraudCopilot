"""
Agent Orchestration Graph
=========================
LangGraph state machine that coordinates the full investigation:

    START
      │
      ▼
  gather_context   -> pulls transaction, device, customer history
      │
      ▼
  run_rule_checks  -> deterministic fraud rule engine
      │
      ▼
  retrieve_history -> RAG over similar past cases
      │
      ▼
  score_risk       -> calls the trained ML risk model
      │
      ▼
  write_report     -> LLM drafts the structured InvestigationReport
      │
      ▼
     END

Each node only reads/writes the shared `AgentState`, so you can test,
swap, or reorder nodes independently. The LLM is only called in the
final node — everything upstream is deterministic and free to run.

Requires: GROQ_API_KEY (or swap ChatGroq for ChatOpenAI/ChatAnthropic).
"""

import os
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.tools.context_tools import get_full_context
from src.tools.rule_engine import check_rules
from src.tools.retrieval import TfidfRetriever, build_query_from_transaction
from src.tools.risk_model_tool import score_transaction
from src.report_schema import InvestigationReport

_retriever = TfidfRetriever()


class AgentState(TypedDict, total=False):
    transaction_id: str
    context: dict
    rule_report: Any
    similar_cases: Any
    risk_score: dict
    report: InvestigationReport


# --- Nodes ---------------------------------------------------------

def gather_context(state: AgentState) -> AgentState:
    ctx = get_full_context(state["transaction_id"])
    return {"context": ctx}


def run_rule_checks(state: AgentState) -> AgentState:
    ctx = state["context"]
    rule_report = check_rules(
        transaction=ctx["transaction"],
        device=ctx["device"],
        customer_history=ctx["customer_history_df"],
    )
    return {"rule_report": rule_report}


def retrieve_history(state: AgentState) -> AgentState:
    ctx = state["context"]
    query = build_query_from_transaction(ctx["transaction"], ctx["device"], state["rule_report"])
    similar = _retriever.retrieve(query, top_k=3)
    return {"similar_cases": similar}


def score_risk(state: AgentState) -> AgentState:
    ctx = state["context"]
    risk = score_transaction(ctx["transaction"], ctx["device"], state["rule_report"])
    return {"risk_score": risk}


REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fraud investigation analyst assistant. You are given structured "
     "evidence about a flagged transaction: transaction details, rule-engine "
     "findings, an ML risk score, and similar historical cases with their "
     "confirmed verdicts. Write a structured investigation report. "
     "Be conservative: only mark 'Fraud' if evidence strongly supports it; "
     "use 'Needs Manual Review' when signals are mixed or weak. "
     "Ground every evidence item in the data provided — do not invent facts."),
    ("human",
     "Transaction: {transaction}\n\n"
     "Device info: {device}\n\n"
     "Customer history summary: {history_summary}\n\n"
     "Rule engine findings: {rule_findings}\n\n"
     "ML risk model output: {risk_score}\n\n"
     "Similar past cases (verdict | note):\n{similar_cases}\n\n"
     "Produce the investigation report now."),
])


def write_report(state: AgentState) -> AgentState:
    llm = ChatGroq(
       model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    structured_llm = llm.with_structured_output(InvestigationReport)

    ctx = state["context"]
    txn = ctx["transaction"]
    similar_text = "\n".join(
        f"- {row.verdict} | {row.case_note}"
        for row in state["similar_cases"].itertuples()
    )

    chain = REPORT_PROMPT | structured_llm
    report = chain.invoke({
        "transaction": txn,
        "device": ctx["device"],
        "history_summary": ctx["customer_history_summary"],
        "rule_findings": state["rule_report"].to_dict(),
        "risk_score": state["risk_score"],
        "similar_cases": similar_text,
    })

    # ensure IDs are correct even if the LLM slips
    report.transaction_id = txn["transaction_id"]
    report.customer_id = txn["customer_id"]

    return {"report": report}


# --- Graph assembly -------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("gather_context", gather_context)
    graph.add_node("run_rule_checks", run_rule_checks)
    graph.add_node("retrieve_history", retrieve_history)
    graph.add_node("score_risk", score_risk)
    graph.add_node("write_report", write_report)

    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context", "run_rule_checks")
    graph.add_edge("run_rule_checks", "retrieve_history")
    graph.add_edge("retrieve_history", "score_risk")
    graph.add_edge("score_risk", "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()


def investigate(transaction_id: str) -> InvestigationReport:
    app = build_graph()
    final_state = app.invoke({"transaction_id": transaction_id})
    return final_state["report"]


if __name__ == "__main__":
    import sys
    txn_id = sys.argv[1] if len(sys.argv) > 1 else "TXN100002"
    report = investigate(txn_id)
    print(report.model_dump_json(indent=2))
