
from rag.embeddings import create_embedding
from rag.vectorstore import VectorStore
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import Reranker

question = "What did the little prince ask the narrator to draw?"
expected_ids = {7, 8}

store = VectorStore.load("storage")
hybrid = HybridRetriever(store)
reranker = Reranker()

query_vector = create_embedding(question)

for k in [30, 100]:

    candidates = hybrid.search(
        question=question,
        query_vector=query_vector,
        k=k,
    )

    print(f"\n{'=' * 60}")
    print(f"HYBRID TOP {k}")
    print(f"Number of candidates: {len(candidates)}")

    for rank, item in enumerate(candidates, start=1):

        chunk_id = item["document"]["chunk_id"]

        if chunk_id in expected_ids:
            print(
                f"EXPECTED CHUNK {chunk_id}: "
                f"hybrid_rank={rank}, "
                f"dense_rank={item['dense_rank']}, "
                f"bm25_rank={item['bm25_rank']}, "
                f"dense_score={item['dense_score']}, "
                f"bm25_score={item['bm25_score']}, "
                f"hybrid_score={item['hybrid_score']:.6f}"
            )

    reranked = reranker.rerank(
        question,
        candidates,
        top_k=len(candidates),
    )

    print("\nEXPECTED CHUNKS AFTER RERANKING")

    found = False

    for rank, item in enumerate(reranked, start=1):

        chunk_id = item["document"]["chunk_id"]

        if chunk_id in expected_ids:
            found = True
            print(
                f"Chunk {chunk_id}: "
                f"rerank_rank={rank}, "
                f"rerank_score={item['rerank_score']:.4f}"
            )

    if not found:
        print("Neither expected chunk was retrieved.")
