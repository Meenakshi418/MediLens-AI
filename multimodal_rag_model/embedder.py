from sentence_transformers import SentenceTransformer
import faiss

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_index(chunk_metadata):
    texts = [item["text"] for item in chunk_metadata]

    embeddings = embedding_model.encode(texts).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)

    return index