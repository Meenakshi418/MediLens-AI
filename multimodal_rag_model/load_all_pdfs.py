import fitz
from doc_registery import save_registry
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import hashlib


def load_all_pdfs(folder_path):

    image_folder = "data/images"
    os.makedirs(image_folder, exist_ok=True)
    image_paths = []
    chunk_metadata = []

    seen_hashes = set() # avoid repeated images

    for pdf_path in os.listdir(folder_path):
        if pdf_path.endswith(".pdf"):
            full_path = os.path.join((folder_path),(pdf_path))
            pdf = fitz.open(full_path)
            page_text = ""
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            for page_number in range(len(pdf)):
                page = pdf[page_number]
                page_text = page.get_text()
                pdf_name = os.path.splitext(pdf_path)[0]
                images = page.get_images(full=True)
                chunks = splitter.split_text(page_text)
                for chunk in chunks:
                    chunk_metadata.append({
                        "text": chunk,
                        "pdf": pdf_name,
                        "page": page_number + 1
                    })


                MIN_DIM = 150  # pixels — logos, signatures, icons are almost always smaller than real content images

                for image_number, image in enumerate(images):
                    xref = image[0]
                    base_image = pdf.extract_image(xref)

                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    if width < MIN_DIM or height < MIN_DIM:
                        continue  # skip logos, signatures, tiny icons

                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)

                    image_path = os.path.join(
                        image_folder,
                        f"{pdf_name}_page_{page_number+1}_img_{image_number+1}.{image_ext}"
                    )

                    with open(image_path, "wb") as file:
                        file.write(image_bytes)

                    image_paths.append(image_path)

    save_registry(chunk_metadata)
    return chunk_metadata, image_paths