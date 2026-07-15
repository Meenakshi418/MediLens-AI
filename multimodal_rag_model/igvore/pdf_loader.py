import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_pdf(pdf_path):

    image_folder = "data/images"
    os.makedirs(image_folder, exist_ok=True)

    pdf = fitz.open(pdf_path)

    text = ""
    image_paths = []

    for page_number in range(len(pdf)):
        page = pdf[page_number]
        text += page.get_text()
        images = page.get_images(full=True)

        for image_number, image in enumerate(images):
            xref = image[0]
            base_image = pdf.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_path = f"{image_folder}/page_{page_number+1}_img_{image_number+1}.{image_ext}"

            with open(image_path, "wb") as file:
                file.write(image_bytes)

            image_paths.append(image_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_text(text)

    return chunks, image_paths