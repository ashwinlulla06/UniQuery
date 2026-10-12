
def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    documents = {}

    for docs in result_lists:
        for rank, doc in enumerate(docs, start = 1):

            chunk_id = doc.metadata['chunk_id']

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)

            documents[chunk_id] = doc

    ranked_chuk_ids = sorted(scores, key=scores.get, reverse=True)

    return [
        documents[chunk_id] 
        for chunk_id in ranked_chuk_ids
    ]
