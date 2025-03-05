import os
from fastapi import FastAPI
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# Load FAISS index & model
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("university_index.faiss")

# Get Hugging Face API Key from Environment Variable
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Load securely

def generate_response_huggingface(prompt):
    """Fetch response from Hugging Face API (Falcon-7B or similar model)."""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 500}}
    response = requests.post(
        "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
        json=payload, headers=headers
    )
    
    if response.status_code == 200:
        return response.json()[0].get("generated_text", "Sorry, I couldn't generate a response.")
    else:
        return f"Error: {response.status_code} - {response.text}"

@app.post("/chatbot/")
async def chatbot(query: str):
    """Searches university knowledge and generates an AI response."""
    
    # Handle missing FAISS index
    if index is None:
        return {"error": "FAISS index not found."}
    
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    
    # Check if we found a valid match
    if I[0][0] == -1:
        best_match_text = "No relevant university data found."
    else:
        best_match_text = " ".join(documents[I[0][0]])  # Ensure `documents` exists

    # Use Hugging Face to generate a response
    ai_response = generate_response_huggingface(f"Question: {query}\nRelevant Info: {best_match_text}\nAnswer:")

    return {"response": ai_response}

# Run FastAPI using Uvicorn with Render's required port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
