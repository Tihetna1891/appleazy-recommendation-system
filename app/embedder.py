# app/embedder.py
from sentence_transformers import SentenceTransformer
import requests
import pdfplumber

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        return ""

    with open("temp_resume.pdf", "wb") as f:
        f.write(response.content)

    full_text = ""
    with pdfplumber.open("temp_resume.pdf") as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() or ""
    
    return full_text.strip()

def embed(text: str):
    return model.encode(text).tolist()

def embed_user(user: dict):
    # Try PDF resume
    if user.get("resume") and user["resume"].endswith(".pdf"):
        text = extract_text_from_pdf(user["resume"])
    elif user.get("cv"):  # fallback to 'cv' if plain text
        text = user["cv"]
    else:
        # Use other fields to generate a vector
        text = f"{user.get('position', '')}, {user.get('preferredIndustry', '')}, {user.get('ethnicity', '')}"

    return embed(text or "No content")
