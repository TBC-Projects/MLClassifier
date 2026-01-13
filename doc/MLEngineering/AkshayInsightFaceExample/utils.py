import numpy as np

def normalize(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def identify(embedding, db_embeddings, labels, threshold=0.5):
    sims = np.dot(db_embeddings, embedding)
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score >= threshold:
        return labels[best_idx], best_score
    return "Unknown", best_score
