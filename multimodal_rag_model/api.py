from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from load_all_pdfs import load_all_pdfs
from embedder import create_index
from image_embedder import create_image_index
from index_manager import (
    save_text_index,
    save_image_index,
    save_metadata,
    save_image_paths
) 

from pydantic import BaseModel
from retriever import retrieve
from llm import ask_llm
from index_manager import (
    load_text_index,
    load_image_index,
    load_metadata,
    load_image_paths
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import Request
from fastapi.staticfiles import StaticFiles

app.mount("/images", StaticFiles(directory="data/images"), name="images")

@app.get("/")
def home():
    return {"message":"backend running"}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    os.makedirs("data/pdfs", exist_ok=True)

    os.makedirs("data/images", exist_ok=True)

    for file in files :
        destination = os.path.join(
            "data/pdfs" ,
            file.filename
        )

        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    chunk_metadata, image_paths = load_all_pdfs("data/pdfs")

    text_index = create_index(chunk_metadata)

    image_index = create_image_index(image_paths)

    save_metadata(chunk_metadata)
    save_image_paths(image_paths)

    save_text_index(text_index)
    save_image_index(image_index)

    return{
        "message": "Document indexed succesfully"
    }     

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(question: Question, request: Request):

    chunk_metadata = load_metadata()
    image_paths = load_image_paths()

    text_index = load_text_index()

    image_index = load_image_index()

    context, retrieved_images = retrieve(
        question.question ,
        text_index,
        image_index,
        chunk_metadata,
        image_paths
    )

    
    answer = ask_llm(
        context,
        retrieved_images,
        question.question
    )
    frontend_images = [
        str(request.base_url) + f"images/{os.path.basename(img)}"
        for img in retrieved_images
    ]

    return{
        "answer": answer,
        "images": frontend_images
    }     
         