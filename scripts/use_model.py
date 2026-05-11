import os
import sys

# Ensure the scripts directory is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(__file__))
from tfidf_baseline import TFIDFBaseline, load_csv


def main():
    data_path = "sample_data/sample_data.csv"
    queries, responses = load_csv(data_path)
    model = TFIDFBaseline()
    model.fit(queries, responses)

    # demo queries (some from the data and one unseen)
    test_queries = [
        "My internet is down, can you help?",
        "I forgot my password",
        "Do you have student discounts?",
        "The app crashes when I open it"
    ]

    for q in test_queries:
        print('\nQuery:', q)
        results = model.retrieve(q, top_k=2)
        for resp, score in results:
            print(f"  score={score:.4f} => {resp}")


if __name__ == '__main__':
    main()
