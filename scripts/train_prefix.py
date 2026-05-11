import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from typing import List, Optional


class TFIDFBaseline:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.corpus: List[str] = []
        self.responses: List[str] = []
        self.tf_idf_matrix = None

    def fit(self, queries: List[str], responses: List[str]):
        """Fit TF-IDF on the list of queries and store responses in parallel."""
        self.corpus = queries
        self.responses = responses
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tf_idf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, top_k: int = 1):
        """Return top_k responses for the query."""
        if self.vectorizer is None or self.tf_idf_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        q_vec = self.vectorizer.transform([query])
        sims = linear_kernel(q_vec, self.tf_idf_matrix).flatten()
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.responses[i], float(sims[i])) for i in top_idx]


def load_csv(path: str):
    df = pd.read_csv(path)
    if 'query' not in df.columns or 'response' not in df.columns:
        raise ValueError("CSV must contain 'query' and 'response' columns")
    return df['query'].astype(str).tolist(), df['response'].astype(str).tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TF-IDF baseline utility")
    parser.add_argument("--data", type=str, default="sample_data/sample_data.csv")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    queries, responses = load_csv(args.data)
    model = TFIDFBaseline()
    model.fit(queries, responses)

    if args.query:
        results = model.retrieve(args.query, top_k=3)
        for r, s in results:
            print(f"score={s:.4f}\tresponse={r}")
    else:
        print(f"Fitted TF-IDF index on {len(queries)} queries. Use --query to retrieve.")
