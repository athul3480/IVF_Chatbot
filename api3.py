from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- Configuration ---
GROQ_API_KEY = "************************************************"
LLM_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.index"
CHUNKS_FILE = "chunks.pkl"

app = FastAPI()

# Enable CORS (Allows your HTML website to talk to this Python script)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace * with your website URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Resources (Run once on startup) ---
print("Loading AI Brain...")
try:
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    client = Groq(api_key=GROQ_API_KEY)
    full_text = "\n\n".join([c["text"] for c in chunks])[:25000] # Safety trim
    print("AI Ready!")
except Exception as e:
    print(f"Error loading files: {e}")

# --- Data Model ---
class ChatRequest(BaseModel):
    message: str

# --- The API Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    
    system_instruction = """You are a helpful IVF assistant.
    1. Answer based on context.
    2. Be PRECISE and COMPACT.
    3. If asked for steps, list them clearly.
    """

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context:\n{full_text}\n\nQuestion: {user_message}"}
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api:app --reload