"""
FastAPI backend with SQLite Authentication. Run with:
    uvicorn app.main:app --reload --port 8000
"""
import os
import uuid
import sqlite3
import hashlib
import secrets
from pathlib import Path
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from app.config import (
    TRAINED_MODEL_PATH, H5AD_PATH, MOESM3_PATH, MOESM5_PATH, PAPER_PDF_PATH,
)
from app.structured_retrieval import StructuredStore
from app.vector_store import PaperVectorStore
from app.model_service import ModelService
from app.rag_report import generate_explanation
from app.chatbot import ChatSession

app = FastAPI(title="BiSpec Pairwise AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 1. Database Setup & Security Utilities ----
DB_PATH = Path(__file__).resolve().parent.parent / "users.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()

# In-memory session token store (token -> email)
ACTIVE_SESSIONS: dict[str, str] = {}

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        'sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000
    ).hex()

# ---- 2. Pipeline Services (Loaded Once) ----
store = StructuredStore(MOESM3_PATH, MOESM5_PATH)
vectors = PaperVectorStore(PAPER_PDF_PATH)
model = ModelService(TRAINED_MODEL_PATH, H5AD_PATH)

_sessions: dict[str, ChatSession] = {}

# ---- 3. Schemas ----
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class PairRequest(BaseModel):
    gene_a: str
    gene_b: str

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

# ---- 4. Authentication Endpoints ----
@app.post("/auth/signup")
def sign_up(req: SignUpRequest):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    
    salt = secrets.token_hex(16)
    p_hash = hash_password(req.password, salt)
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (email, password_hash, salt) VALUES (?, ?, ?)",
                (req.email.lower().strip(), p_hash, salt)
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
    
    token = secrets.token_urlsafe(32)
    ACTIVE_SESSIONS[token] = req.email.lower().strip()
    return {"token": token, "email": req.email.lower().strip(), "message": "Account created successfully."}

@app.post("/auth/signin")
def sign_in(req: SignInRequest):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, salt FROM users WHERE email = ?", (req.email.lower().strip(),))
        row = cursor.fetchone()
        
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
        
    stored_hash, salt = row
    if hash_password(req.password, salt) != stored_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
        
    token = secrets.token_urlsafe(32)
    ACTIVE_SESSIONS[token] = req.email.lower().strip()
    return {"token": token, "email": req.email.lower().strip(), "message": "Signed in successfully."}

@app.get("/auth/me")
def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    email = ACTIVE_SESSIONS.get(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return {"email": email}

# ---- 5. Application & Static Routes ----
@app.get("/", response_class=FileResponse)
def serve_index():
    base_dir = Path(__file__).resolve().parent.parent
    index_file = base_dir / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>index.html not found in frontend/</h2>", status_code=404)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PairRequest):
    ctx = store.lookup(req.gene_a, req.gene_b)
    model_out = model.predict(req.gene_a, req.gene_b, ctx)
    return {**ctx, **model_out}

@app.post("/report")
def report(req: PairRequest):
    ctx = store.lookup(req.gene_a, req.gene_b)
    model_out = model.predict(req.gene_a, req.gene_b, ctx)
    chunks = vectors.retrieve(f"{req.gene_a} {req.gene_b} target combination")
    explanation = generate_explanation(ctx, model_out, chunks)
    return {**ctx, **model_out, "explanation": explanation, "n_chunks_used": len(chunks)}

@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession(store, vectors, model)
    reply = _sessions[session_id].ask(req.message)
    return {"session_id": session_id, "reply": reply}