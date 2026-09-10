# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
from dotenv import load_dotenv
from google import genai

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
embedder = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI(title="Ethiopian Secondary Curriculum AI Tutor")

class QueryRequest(BaseModel):
    question: str
    is_online: bool = True
    grade: int | None = None
    subject: str | None = None

def fetch_web_context(query: str) -> str:
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return ""
        snippets = [f"- {item.get('title')}: {item.get('body')}" for item in results]
        return "\n".join(snippets)
    except Exception:
        return ""

@app.get("/")
def root():
    return {"status": "Ethiopian Curriculum AI Tutor API is running"}

@app.post("/api/ask")
def ask_question(request: QueryRequest):
    # 1. Generate semantic embedding vector for the question
    query_vector = embedder.encode(request.question).tolist()

    # 2. Prepare RPC arguments matching your Supabase function signature
    rpc_params = {
        'query_embedding': query_vector,
        'match_threshold': 0.1,  # Lowers threshold for high recall across textbook terms
        'match_count': 5,        # Top 5 most relevant pages
        'target_grade': request.grade,
        'target_subject': request.subject
    }

    # 3. Perform Vector Similarity Search in Supabase
    try:
        results = supabase.rpc('match_textbook_chunks', rpc_params).execute()
        matches = results.data or []
    except Exception as e:
        matches = []

    textbook_context = ""
    sources = []

    if matches:
        for match in matches:
            sources.append({
                "grade": match.get('grade'),
                "subject": match.get('subject'),
                "page_number": match.get('page_number')
            })
            textbook_context += f"\n--- [Grade {match.get('grade')} {match.get('subject')} - Page {match.get('page_number')}] ---\n{match.get('content')}\n"

    # 4. Fetch optional web search info if offline retrieval returns empty
    web_info = fetch_web_context(request.question) if (request.is_online and not textbook_context) else ""

    # 5. Formulate Context-Aware Prompt
    if textbook_context:
        prompt = f"""You are an expert AI tutor for the Ethiopian Secondary Curriculum (Grades 9-12).
Answer the user's question clearly, thoroughly, and accurately based strictly on the retrieved textbook context below.
Always cite the Grade, Subject, and Page numbers at the bottom of your answer.

TEXTBOOK CONTEXT:
{textbook_context}

USER QUESTION: {request.question}"""
    else:
        prompt = f"""You are an expert AI tutor for Ethiopian secondary students.
The requested topic was not found directly in the Grade 9-12 textbook database.
Provide a clear, helpful answer using general knowledge and web information. Briefly mention that the details come from general educational resources rather than the uploaded textbook pages.

WEB SEARCH CONTEXT:
{web_info if web_info else "None"}

USER QUESTION: {request.question}"""

    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in your .env file.")

    # 6. Generate answer via Gemini 3.6 Flash
    try:
        response = gemini_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=str(prompt)
        )
        return {
            "answer": response.text,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")