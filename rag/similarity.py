from sklearn.metrics.pairwise import cosine_similarity


def find_best_match(query_vector, chunk_vectors, chunks):

    scores = cosine_similarity([query_vector], chunk_vectors)[0]

    best_index = scores.argmax()

    return chunks[best_index], scores[best_index]
