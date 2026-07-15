import fitz #PyMuPDF


pdf = fitz.open("sample.pdf")
print("Pages:", len(pdf))# count pages

page = pdf[0]
text = page.get_text()
# print(text)

# for page_num in range(len(pdf)):
#     page = pdf[page_num]
#     text = page.get_text()
#     print(f"\npage{page_num + 1}")
#     print(text)

all_text = ""
for page in pdf:    #creates a one large string of text
    all_text += page.get_text()

# print(all_text)
# print(type(all_text))
# print(len(all_text))

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
#chunk_size=100 → each chunk has about 100 characters
#chunk_overlap=20 → the next chunk repeats the last 20 characters(in case a sentence gets cut in half overlap allows it to remain completed by using prev 20 char)

chunks = text_splitter.split_text(all_text)
# print(type(chunks))# list
# print(len(chunks))

# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}")
#     print(chunk)

# downloading pretrained embedding model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#Convert chunks into embedding
embeddings = embedding_model.encode(chunks)

# print(type(embeddings)) # numpy.ndarray- numpy array
# print(embeddings.shape)
# print(embeddings[0][:10])

# faiss - a vector db table
import faiss

print(embeddings.dtype)# to check for float32 as faiss needs float32 not python lists

dimension = embeddings.shape[1]# embedding (chunks , numbers) shape[1] = numbers
# how many numbers in one embedding(1 embedding = 1 chunk)
print(dimension) # 384 always

# create vector db
index = faiss.IndexFlatL2(dimension)
# Index = db , Flat = no compression , L2 = dis formula

# store in database
index.add(embeddings)
print(index.ntotal)



import numpy as np

query = "what is the solution for the given questionin pdf?"

query_embedding = embedding_model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")
#FAISS is very strict:it does NOT accept Python lists it expects NumPy float32 arrays: list → numpy array → float32
print(query_embedding.shape)

k=3 # decide how many result. top 3 result

# faiss search
distances, indices = index.search(query_embedding, k)

# Then it returns:
# indices = [[1, 2, 5]] - is a 2d array where all top 3 are stored for every query this only has one list as there is one query [[1,2,3],[2,4,7]]
# distances = [[0.20, 0.35, 0.41]]

print(indices)

retrieved_chunks = []
for i in indices[0]: # indice 0 represents query 1 if i=there were more querys more get executed
    retrieved_chunks.append(chunks[i])

# merge all top 3 chunks into one
context = "\n\n".join(retrieved_chunks)
print(context)

# LLM Model
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

prompt = f"""
You are a helpful assistant.

Answer ONLY using the information provided in the context.
If the answer is not present, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

response = model.generate_content(prompt)
print(response.text)