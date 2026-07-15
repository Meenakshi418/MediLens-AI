import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

image_folder = "data/images"
os.makedirs(image_folder, exist_ok=True)
# to create image folder incase it doesnt exists prvents errors

image_paths = [] # empty list to store all images
pdf = fitz.open("sample.pdf")

for page_number in range(len(pdf)):
    page = pdf[page_number]
    images = page.get_images(full=True) # info abt image i.e. where it is

    for image_number, image in enumerate(images): # if a page has diag & logo then img_num = 0 & 1
        xref = image[0] # every image  has a id to extract actual img
        base_image = pdf.extract_image(xref) # now we get the binary data and type info

        #seperate the info   
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        # create a filename = page_1_img_1.png instead of img1.png
        image_path = f"{image_folder}/page_{page_number+1}_img_{image_number+1}.{image_ext}"
         
        # now the img exists physicaly in comp
        with open(image_path, "wb") as file:
            file.write(image_bytes)

        image_paths.append(image_path)

        print(image_paths) 


from transformers import CLIPProcessor, CLIPModel # understand imge and text data has 512 numvbers
#CLIPModel - This is the pretrained AI model.CLIPProcessor - Before the model can read an image,the image must be resized, normalized, converted into tensors, etc.

from PIL import Image # opens images when they are in .png form as that isds just a folder

import torch 
#CLIP works on PyTorch tensors.The processor returns tensors.

import faiss
# stroe image vectors

import numpy as np
# sometimes when clip returns tensors . faiss want numpy array so this does conversion

# downloads pretrained model
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)
# loads the preprocessing rules
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
#Sets the model to evaluation mode.That means "I'm not training the model. I'm only using it for predictions."
clip_model.eval()

image_embeddings = [] # store all img embeddings


for image_path in image_paths:
    image = Image.open(image_path)  # opens the image 

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ) # resize image , normalize pixel values , convert  to tensor

    with torch.no_grad():# not trainig clip using it so pytorch skips storing gradients
        embeddings = clip_model.get_image_features(**inputs)# vectore values of img
        embedding = embedding.cpu().numpy()#convert to np array . cant access gpu memory so converts to cpu

        embedding = embedding.squeeze() #Remove the extra dimension. Right now the shape is(1, 512)because you processed one image.We only want(512,)

        image_embeddings.append(embedding)# prcessing top 3 images

    embeddings = np.array(image_embeddings).astype("float32")   #Now its shape becomes (number_of_images, 512)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print(index)

    # Search
# ----------------------------------------------------

k = 3
# retrieve top 3 similar images

distances, indices = index.search(
    query_embedding,
    k
)

# indices tells which images matched
# distances tells how close they are

print(indices)

retrieved_images = []

# indices[0] because we searched only one query
for i in indices[0]:

    # convert index into actual image path
    retrieved_images.append(image_paths[i])

print(retrieved_images)

# ----------------------------------------------------
# Gemini
# ----------------------------------------------------

import google.generativeai as genai
from PIL import Image
import os

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

opened_images = []

# Gemini needs actual images not file paths
for image_path in retrieved_images:

    image = Image.open(image_path)

    opened_images.append(image)

prompt = """
Answer using only the retrieved images.
"""

# prompt + all retrieved images are sent together
response = model.generate_content(
    [prompt] + opened_images
)

print(response.text)