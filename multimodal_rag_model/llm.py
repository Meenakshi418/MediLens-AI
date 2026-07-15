import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY4"))

model = genai.GenerativeModel("gemini-2.5-flash")


def ask_llm(context, retrieved_images, query):

    prompt = f"""
You are a medical document assistant helping a patient understand their own report.

Rules:
- Answer using ONLY the context and images below. Do not use outside medical knowledge to fill gaps.
- When you do answer, quote or reference the specific value/finding from the document, then briefly explain what it means in plain language.
- Do not speculate about conditions, diagnoses, or values the document does not mention.

Formatting Rules:
- Formatting Rules:
- Maximum 5 bullet points.
- Maximum 80 words.
- Each bullet must follow this format:
  Finding: Value – Short implication.
- Keep each bullet under 12 words.
- Do not explain reference ranges.
- Do not explain medical terms unless asked.
- If multiple findings indicate the same condition, combine them.
- If the user requests a detailed explanation, you may ignore the word limit.

Context:
{context}

Question:
{query}

Answer:
"""
    opened_images = []
    for image_path in retrieved_images:
        with Image.open(image_path) as image:
            opened_images.append(image.copy())
   

    response = model.generate_content([prompt] + opened_images)

    return response.text

def ask_llm_no_context(query):
    prompt = f"""The user asked: "{query}"

No relevant information on this topic was found in their uploaded documents.

Reply in a short, direct, natural way — as if answering their yes/no question — 
but make clear the reports simply don't mention it. 
For example: "No, the reports don't mention any patient having diabetes." or "the reports don"t indicate to pneumonia".
make sure to mention the illness or query they are trying to clarify in the ans and never answer "I could not find the answer in the provided document." this shows that u are unaware of it so never mention it instead say "the information for issue / query is not mentioned in document " never say i couldnt say it doesnt mention .

Do NOT claim the condition was tested for or ruled out. Only state that it isn't mentioned.
"""
    response = model.generate_content(prompt)
    return response.text