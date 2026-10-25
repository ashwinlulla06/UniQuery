from retrieving_and_answering import semantic_retriever, bm25_retriever, rrf
from rag_evals.test_question import get_test_questions, TestQuestion
from pydantic import BaseModel, Field
import math
from reranking.reranker import rerank_documents
import config

"""
Here we are using different retrieval metrics to check the performance of the retrieval
"""

DB_NAME = config.DB_NAME
class RetrievalEval(BaseModel):

    mrr_score: float = Field(description="MRR score based on the relevance of the chunks")

    ndcg_score: float = Field(description="nDCG score based on the relevance of the chunks")

    recall_score: float = Field(description="Recall@K score based on the relevance of the retrieved chunks")

    hit_score: float = Field(description="Hit@K score indicating whether at least one relevant chunk was retrieved")

    precision_score: float = Field(description="Precision@K score based on the relevance of the retrieved chunks")



def calculate_mrr(retrieved_docs : list, relevant_chunk_ids : list) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc.metadata.get('chunk_id') in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0

def calculate_dcg(relevance : list, k : int = 5):
    dcg = 0.0
    for i in range(min(k, len(relevance))):
        dcg += relevance[i] / math.log2(i+2)
    return dcg

def calculate_ndcg(rel_chunk_ids : list, retrieved_docs : list, k : int = 5):

    relevance = [
        1 if doc.metadata['chunk_id'] in rel_chunk_ids else 0
        for doc in retrieved_docs
    ]

    dcg = calculate_dcg(relevance, k)

    num_relevant = min(len(rel_chunk_ids), k)  # Getting total number of relevant chunks for the questions

    ideal_relevance = [1] * num_relevant + [0] * (k - num_relevant)  # making the ideal_relevance looks like: [1, 1, 1, 0, 0] if total relevant chunks are 3!

    idcg = calculate_dcg(ideal_relevance, k)

    ndcg = dcg / idcg if idcg > 0 else 0

    return ndcg   


def calculate_recall(rel_chunk_ids, retrieved_docs):

    total_rel_chunks_retrieved = sum(
        1 if doc.metadata['chunk_id'] in rel_chunk_ids else 0 
        for doc in retrieved_docs
    )

    recall = total_rel_chunks_retrieved / len(rel_chunk_ids) if len(rel_chunk_ids) > 0 else 0

    return  recall

def calculate_hit(rel_chunk_ids, retrieved_docs):
    
    for doc in retrieved_docs:
        if doc.metadata['chunk_id'] in rel_chunk_ids:
            return 1

    return 0


def calculate_precision(rel_chunk_ids, retrieved_docs):

    if retrieved_docs[0].metadata['chunk_id'] in rel_chunk_ids:
        return 1
    return 0
    
    
def evaluate_retrieval(test : TestQuestion, k : int = 5) -> RetrievalEval:

    semantic_docs = semantic_retriever.fetch_documents(test.question)
    bm25_docs = bm25_retriever.keyword_retriever(test.question)

    retrieved_docs = rrf.reciprocal_rank_fusion([bm25_docs, semantic_docs])

    reranked_docs = rerank_documents(test.question, retrieved_docs)

    mrr_score = calculate_mrr(reranked_docs, test.relevant_chunk_ids)
    ndcg_score = calculate_ndcg(test.relevant_chunk_ids, reranked_docs, k)
    recall_score = calculate_recall(test.relevant_chunk_ids, reranked_docs)
    hit_score = calculate_hit(test.relevant_chunk_ids, reranked_docs)
    precision_score = calculate_precision(test.relevant_chunk_ids, reranked_docs)

    return RetrievalEval(
        mrr_score = mrr_score,
        ndcg_score = ndcg_score,
        recall_score=recall_score,
        hit_score=hit_score,
        precision_score=precision_score
    )

def evaluate_all_retrieval():
    """Evaluate all retrieval tests."""
    tests = [
        test
        for test in get_test_questions()
        if test.category != 'unanswerable'
    ]
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress
