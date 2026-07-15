from embedder import embedding_model
from clip_utils import clip_model, clip_processor
import torch
import numpy as np
import faiss

def retrieve(query, text_index, image_index, chunk_metadata, image_paths, k=10):
    
    

    # Encode query for text index
    text_query_embedding = embedding_model.encode([query]).astype("float32")
    faiss.normalize_L2(text_query_embedding)

    text_scores, text_indices = text_index.search(text_query_embedding, k)

    TEXT_THRESHOLD = 0.35  # tune this — cosine similarity, higher = more similar

    retrieved_metadata = [
        chunk_metadata[i]
        for score, i in zip(text_scores[0], text_indices[0])
        if i < len(chunk_metadata) and score > TEXT_THRESHOLD
    ]

    # Tokenize query for CLIP image index matching
    inputs = clip_processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        image_query_embedding = clip_model.get_text_features(**inputs)

    # Safely unpack the tensor depending on HF version
    if hasattr(image_query_embedding, "text_embeds") and image_query_embedding.text_embeds is not None:
        image_query_embedding = image_query_embedding.text_embeds
    elif hasattr(image_query_embedding, "pooler_output") and image_query_embedding.pooler_output is not None:
        image_query_embedding = image_query_embedding.pooler_output
    elif hasattr(image_query_embedding, "image_embeds") and image_query_embedding.image_embeds is not None:
        image_query_embedding = image_query_embedding.image_embeds

    # Convert to numpy exactly ONCE and make it float32
    if isinstance(image_query_embedding, torch.Tensor):
        image_query_embedding = image_query_embedding.cpu().numpy()
        
    image_query_embedding = image_query_embedding.astype("float32")
    faiss.normalize_L2(image_query_embedding) #new

    # Search text and image indexes
    text_distances, text_indices = text_index.search(text_query_embedding, k)

    #new
    if image_index.ntotal == 0:
        retrieved_images = []
    else:
        
        # k = min(k, len(image_paths))
        k=2

        image_distances, image_indices = image_index.search(
            image_query_embedding,
            k
        )

        retrieved_metadata = [
        chunk_metadata[i]
            for i in text_indices[0]
            if i < len(chunk_metadata)
        ]

        retrieved_images = []

        # THRESHOLD = 10  # adjust later

        # for dist, idx in zip(image_distances[0], image_indices[0]):
        #     if idx < len(image_paths) and dist < THRESHOLD:
        #         retrieved_images.append(image_paths[idx])

        SIMILARITY_THRESHOLD = 0.2   # now higher = more similar, tune this instead

        for score, idx in zip(image_distances[0], image_indices[0]):
            if idx < len(image_paths) and score > SIMILARITY_THRESHOLD:
                retrieved_images.append(image_paths[idx])

        context = ""

        for item in retrieved_metadata:
            context += (
                f"[{item['pdf']}.pdf | Page {item['page']}]\n"
                f"{item['text']}\n\n"
            )
    return context, retrieved_images