# retrieval_models.py
from collections import defaultdict, Counter
import math

def bm25_score(query, inverted_index, k=1.4, b=0.7):
    """
    Compute BM25 scores for a given query against all documents in the inverted index.

    Parameters:
    - query: Query object
    - inverted_index: InvertedIndex object
    - k: BM25 k parameter (default=1.5)
    - b: BM25 b parameter (default=0.75)

    Returns:
    - scores: Dictionary mapping doc_id to BM25 score
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score  # Incorporate term weight
    return scores

from tqdm import tqdm  # Import tqdm

def mixture_model_prf(query, inverted_index, top_n=10, num_expansion_terms=10, max_iterations=50, convergence_threshold=1e-5, lambda_init=0.5):
    """
    Perform Pseudo Relevance Feedback using the Mixture Model.

    Parameters:
    - query: Query object
    - inverted_index: InvertedIndex object
    - top_n: Number of top documents to consider as pseudo-relevant (default=10)
    - num_expansion_terms: Number of terms to select for query expansion (default=10)
    - max_iterations: Maximum number of EM iterations (default=50)
    - convergence_threshold: Threshold for convergence (default=1e-5)
    - lambda_init: Initial value for lambda (default=0.5)

    Returns:
    - updated_query: Query object with expanded terms
    """
    # Step 1: Initial Retrieval
    initial_scores = bm25_score(query, inverted_index)
    ranked_docs = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc_id for doc_id, _ in ranked_docs[:top_n]]

    # Step 2: Collect term frequencies in top documents
    term_counts = Counter()
    total_top_terms = 0
    for doc_id in top_docs:
        tokens = inverted_index.doc_tokens[doc_id]
        term_counts.update(tokens)
        total_top_terms += len(tokens)

    # Step 3: Compute background probabilities P(w | C)
    vocab = set(term_counts.keys())
    min_prob = 1e-10  # Small probability to avoid division by zero
    P_bg = {term: max(inverted_index.compute_collection_prob(term), min_prob) for term in vocab}

    # Step 4: Initialize P(w | R) and lambda
    P_w_R = {term: count / total_top_terms for term, count in term_counts.items()}
    lambda_val = lambda_init

    # Step 5: EM Algorithm with Progress Bar
    for iteration in tqdm(range(max_iterations), desc="EM Iterations", unit="iter"):
        P_w_R_old = P_w_R.copy()

        # E-step: Compute gamma(w) for each term
        gamma = {}
        for term in vocab:
            numerator = lambda_val * P_w_R[term]
            denominator = numerator + (1 - lambda_val) * P_bg[term]
            if denominator == 0:
                gamma[term] = 0
            else:
                gamma[term] = numerator / denominator

        # M-step: Update P(w | R) and lambda
        # Update P(w | R)
        P_w_R_denominator = sum(gamma[term] * term_counts[term] for term in vocab)
        for term in vocab:
            count = term_counts[term]
            gamma_w = gamma[term]
            P_w_R[term] = (gamma_w * count) / P_w_R_denominator if P_w_R_denominator > 0 else 0

        # Update lambda
        lambda_numerator = sum(gamma[term] * term_counts[term] for term in vocab)
        lambda_denominator = total_top_terms
        lambda_val = lambda_numerator / lambda_denominator if lambda_denominator > 0 else lambda_init

        # Check convergence
        diff = sum(abs(P_w_R[term] - P_w_R_old[term]) for term in vocab)

        # Optionally, update progress bar description
        # tqdm.write(f"Iteration {iteration+1}: Convergence diff = {diff:.6f}")

        if diff < convergence_threshold:
            break

    # Step 6: Select top terms for query expansion
    expansion_candidates = sorted(P_w_R.items(), key=lambda x: x[1], reverse=True)
    expansion_terms = {}
    count = 0
    for term, prob in expansion_candidates:
        if term not in query.tokens:
            expansion_terms[term] = prob
            count += 1
        if count >= num_expansion_terms:
            break

    # Normalize expansion term weights
    max_prob = max(expansion_terms.values()) if expansion_terms else 1
    expansion_terms = {term: prob / max_prob for term, prob in expansion_terms.items()}

    # Step 7: Update the query
    query.expand(expansion_terms)

    return query