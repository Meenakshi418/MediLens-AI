from clip_utils import clip_model, clip_processor
from PIL import Image
import torch
import faiss
import numpy as np

def create_image_index(image_paths):
    #new
    if not image_paths:
        return faiss.IndexFlatIP(512)  # empty index, same dimension as CLIP output
    
    image_embeddings = []
    for image_path in image_paths:
        image = Image.open(image_path)
        inputs = clip_processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            # Extract features specifically using the image method
            outputs = clip_model.get_image_features(**inputs)
            
            # Unpack the tensor if HF returned it wrapped in an output class
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                embedding = outputs.image_embeds
            else:
                embedding = outputs

            # Now it's safe to send to CPU and convert to numpy
            embedding = embedding.cpu().numpy().squeeze() 
            
            # Track it
            image_embeddings.append(embedding)

    # embeddings = np.array(image_embeddings).astype("float32")

    # dimension = embeddings.shape[1]
    # index = faiss.IndexFlatL2(dimension)
    # index.add(embeddings)
    
    embeddings = np.array(image_embeddings).astype("float32")
    faiss.normalize_L2(embeddings)          # add this line

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)    # inner product = cosine similarity on normalized vectors
    index.add(embeddings)

    return index