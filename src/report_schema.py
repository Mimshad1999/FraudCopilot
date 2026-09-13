"""
Structured Report Schema
=========================
Forces the LLM's final output into a predictable, parseable shape —
this is what makes the report usable in a real UI/workflow instead
of being a wall of unstructured text.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class EvidenceItem(BaseModel):
    source: str = Field(description="Which signal this evidence came from, e.g. 'rule_engine', 'risk_model', 'past_cases', 'device_check'")
    finding: str = Field(description="One-sentence description of the finding")


class InvestigationReport(BaseModel):
    transaction_id: str
    customer_id: str
    verdict: Literal["Fraud", "Likely Fraud", "Needs Manual Review", "Legitimate"]
    confidence: float = Field(ge=0, le=1, description="Model/agent confidence in the verdict, 0-1")
    summary: str = Field(description="2-3 sentence plain-English summary of the case for a human reviewer")
    evidence: List[EvidenceItem]
    recommended_action: str = Field(description="e.g. 'Freeze account and contact customer', 'No action needed', 'Escalate to senior analyst'")
    reasoning: str = Field(description="Short chain-of-reasoning explaining how the verdict was reached from the evidence")
