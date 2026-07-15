import os
import faiss
import json


def save_text_index(text_index):
    faiss.write_index(text_index, "indexes/text.index")
    

def load_text_index():
    text_index = faiss.read_index("indexes/text.index")
    return text_index


def save_image_index(image_index):
    faiss.write_index(image_index, "indexes/image.index")

def load_image_index():
    image_index = faiss.read_index("indexes/image.index")
    return image_index


def save_image_paths(image_paths):
    with open("indexes/image_path.json", "w") as file:
        json.dump(image_paths, file)

def load_image_paths():
    with open("indexes/image_path.json") as file:
        image_paths = json.load(file)
    return image_paths


def save_metadata(chunk_metadata):
    with open("indexes/chunks.json", "w") as file:
        json.dump(chunk_metadata, file)

def load_metadata():
    with open("indexes/chunks.json") as file:
        chunk_metadata = json.load(file)

    return chunk_metadata