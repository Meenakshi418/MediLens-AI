from load_all_pdfs import load_all_pdfs
from embedder import create_index
from image_embedder import create_image_index
from retriever import retrieve
from llm import ask_llm , ask_llm_no_context
from index_manager import save_image_index, save_image_paths, save_metadata, save_text_index, load_image_index, load_image_paths, load_metadata, load_text_index
import os

os.makedirs("indexes", exist_ok=True)

if all([
    os.path.exists("indexes/text.index"),
    os.path.exists("indexes/image.index"),
    os.path.exists("indexes/chunks.json"),
    os.path.exists("indexes/image_path.json")
]):
    text_index = load_text_index()
    image_index = load_image_index()
    chunk_metadata = load_metadata()
    image_paths = load_image_paths()

else:
    chunk_metadata, image_paths = load_all_pdfs("data/pdfs")
    save_metadata(chunk_metadata)
    save_image_paths(image_paths)

    text_index = create_index(chunk_metadata)
    save_text_index(text_index)

    image_index = create_image_index(image_paths)
    save_image_index(image_index)



while True:
    query = input("\nAsk (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    context, retrieved_images = retrieve( query, text_index, image_index, chunk_metadata, image_paths)

    if not context.strip():
        answer = ask_llm_no_context(query)
    else:
        answer = ask_llm(context, retrieved_images, query)

    print("\nAnswer:")
    print(answer)