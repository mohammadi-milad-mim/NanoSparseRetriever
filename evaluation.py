# evaluation.py
from collections import defaultdict
import math
import numpy as np

class EvaluationMetrics:
    @staticmethod
    def precision_at_k(retrieved_docs, relevant_docs, k):
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = set(retrieved_at_k) & set(relevant_docs)
        return len(relevant_retrieved) / k

    @staticmethod
    def recall(retrieved_docs, relevant_docs):
        relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
        return len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
    
    @staticmethod
    def recall_at_k(retrieved_docs, relevant_docs, k):
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = set(retrieved_at_k) & set(relevant_docs)
        return len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

    @staticmethod
    def average_precision(retrieved_docs, relevant_docs):
        if not relevant_docs:
            return 0

        relevant_retrieved = 0
        cumulative_precision = 0
        relevant_docs_set = set(relevant_docs)

        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs_set:
                relevant_retrieved += 1
                cumulative_precision += relevant_retrieved / i
        
        return cumulative_precision / len(relevant_docs)

    @staticmethod
    def mean_average_precision(all_retrieved_docs, all_relevant_docs):
        total_ap = 0
        num_queries = len(all_retrieved_docs)

        if num_queries == 0:
            return 0

        for query_id, retrieved_docs in all_retrieved_docs.items():
            relevant_docs = all_relevant_docs.get(query_id, set())
            total_ap += EvaluationMetrics.average_precision(retrieved_docs, relevant_docs)
        
        return total_ap / num_queries

    @staticmethod
    def ndcg(retrieved_docs, relevant_docs, k):
        if not relevant_docs:
            return 0
        
        # Compute DCG
        dcg = 0.0
        relevant_docs_set = set(relevant_docs)
        for i in range(min(k, len(retrieved_docs))):
            doc_id = retrieved_docs[i]
            if doc_id in relevant_docs_set:
                dcg += 1 / math.log2(i + 2)

        # Compute IDCG (Ideal DCG)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_docs))))

        return dcg / idcg if idcg > 0 else 0

def load_relevance_judgments(file_path):
    """
    Load the golden data file and return a mapping from query IDs to sets of relevant document IDs.
    """
    relevance_judgments = defaultdict(set)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Assuming the file format is: QueryID 0 DocID RelevanceScore
            parts = line.strip().split()
            if len(parts) == 4:
                query_id = parts[0]
                doc_id = parts[2]
                relevance_score = int(parts[3])
                if relevance_score > 0:
                    relevance_judgments[query_id].add(doc_id)
    return relevance_judgments


def evaluate_scoring_method(inverted_index, scoring_method, queries, relevance_judgments, params={}):
    """
    Evaluate a given scoring method for all queries and compute relevant metrics.
    
    Parameters:
    - scoring_method: A function that computes scores for a query, such as bm25_score.
    - queries: A list of queries to evaluate.
    - relevance_judgments: A dictionary of relevance judgments for the queries.
    - params: A dictionary of parameters for the scoring method (e.g., {'k': 1.5, 'b': 0.75} for BM25).
    
    Returns:
    - results: A dictionary containing MAP, Precision@5, Recall@10, and NDCG@5 for the method.
    """
    all_retrieved_docs = {}
    precision_at_5_scores = []
    recall_scores = []
    ndcg_scores = []
    recall_at_10_scores = []
    
    for query in queries:
        # Compute scores for the current query by passing params to the scoring method
        scores = scoring_method(query, inverted_index, **params)
        
        # Rank documents based on scores
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Retrieve document IDs
        retrieved_docs = [doc_id for doc_id, _ in ranked_docs]
        
        # Store retrieved documents for the current query
        all_retrieved_docs[query.query_id] = retrieved_docs
        
        # Get relevant docs for the current query
        relevant_docs = relevance_judgments.get(query.query_id, set())

        # Precision@5
        precision_at_5 = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_docs, 5)
        precision_at_5_scores.append(precision_at_5)

        # Recall
        recall = EvaluationMetrics.recall(retrieved_docs, relevant_docs)
        recall_scores.append(recall)
        
        # Recall@10
        recall_at_10 = EvaluationMetrics.recall_at_k(retrieved_docs, relevant_docs, 10)
        recall_at_10_scores.append(recall_at_10)

        # NDCG@5
        ndcg = EvaluationMetrics.ndcg(retrieved_docs, relevant_docs, 5)
        ndcg_scores.append(ndcg)

    # Evaluate Mean Average Precision (MAP)
    map_score = EvaluationMetrics.mean_average_precision(
        all_retrieved_docs, relevance_judgments
    )

    # Compute average metrics across all queries
    avg_precision_at_5 = float(np.mean(precision_at_5_scores))
    avg_recall = float(np.mean(recall_scores))
    avg_recall_at_10 = float(np.mean(recall_at_10_scores))
    avg_ndcg = float(np.mean(ndcg_scores))

    # Store results
    results = {
        'MAP': round(map_score, 4),
        'Precision@5': round(avg_precision_at_5, 4),
        'Recall@10': round(avg_recall_at_10, 4),
        'nDCG@5': round(avg_ndcg, 4)
    }

    return results