"""
Similar Past-Case Retrieval (RAG)
==================================
Retrieves the most similar historical case notes for a given
transaction, so the History Agent can reason by precedent
("we've seen this pattern before, and it was confirmed fraud").

Uses TF-IDF + cosine similarity by default so this runs fully
offline with zero API calls or model downloads — good for
development and demos. For production quality, swap
`TfidfRetriever` for a proper embedding-based retriever
(see the commented `EmbeddingRetriever` stub below) backed by
Chroma/FAISS + sentence-transformers or an OpenAI/Groq embedding
endpoint. The public interface (`retrieve`) stays identical either
way, so nothing else in the agent graph needs to change.
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class TfidfRetriever:
    def __init__(self, csv_path=None):
        csv_path = csv_path or DATA_DIR / "past_cases.csv"
        self.cases = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.case_vectors = self.vectorizer.fit_transform(self.cases["case_note"])

    def retrieve(self, query_text: str, top_k: int = 3) -> pd.DataFrame:
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.case_vectors).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = self.cases.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        return results[["case_id", "verdict", "case_note", "similarity"]]


def build_query_from_transaction(txn: dict, device: dict, rule_report=None) -> str:
    """Turns a flagged transaction into a natural-language query so
    the retriever can find textually similar past cases."""
    parts = [
        f"Transaction of amount {txn['amount_inr']} at merchant {txn['merchant']} in city {txn['city']}.",
        f"Device type: {device.get('device_type', 'unknown')}.",
    ]
    if device.get("is_new_device_for_customer"):
        parts.append("This is a new/unrecognized device for the customer.")
    if rule_report is not None:
        for r in rule_report.triggered_rules:
            parts.append(r.detail)
    return " ".join(parts)


# --- Upgrade path (commented out — requires internet / API key) ---
#
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings  # or OpenAIEmbeddings
#
# class EmbeddingRetriever:
#     def __init__(self, csv_path=None):
#         csv_path = csv_path or DATA_DIR / "past_cases.csv"
#         cases = pd.read_csv(csv_path)
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         self.store = Chroma.from_texts(
#             texts=cases["case_note"].tolist(),
#             embedding=embeddings,
#             metadatas=cases[["case_id", "verdict"]].to_dict("records"),
#         )
#
#     def retrieve(self, query_text: str, top_k: int = 3):
#         return self.store.similarity_search(query_text, k=top_k)


if __name__ == "__main__":
    # quick smoke test
    retriever = TfidfRetriever()
    sample_query = "Transaction of amount 45000 at merchant Crypto Exchange XYZ in city Dubai. New device."
    results = retriever.retrieve(sample_query, top_k=3)
    print(results.to_string())
