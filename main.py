import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
import google.generativeai as genai

# Load hidden environment variables from .env file
load_dotenv()

app = FastAPI(title="Ethiopian Curriculum AI RAG API")

# Add CORS Middleware to allow requests from Streamlit Cloud & Android WebViews
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Credentials securely from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Load Local Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def resolve_gemini_model():
    """Detects and returns the appropriate Gemini model ID for your API key."""
    candidates = [
        "gemini-3.6-flash",
        "models/gemini-3.6-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    try:
        available = [
            m.name
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]

        for candidate in candidates:
            formatted_name = (
                candidate
                if candidate.startswith("models/")
                else f"models/{candidate}"
            )
            if formatted_name in available or candidate in available:
                return candidate

        if available:
            return available[0]
    except Exception as err:
        print(f"Warning during model inspection: {err}")

    return "gemini-3.6-flash"


class QueryRequest(BaseModel):
    query: str
    grade: int | None = None
    subject: str | None = None
    top_k: int = 4


@app.get("/")
def health_check():
    """Root health check for Render container monitoring."""
    return {"status": "ok", "message": "Ethiopian Curriculum AI API is running!"}


@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    try:
        # 1. Generate query vector embedding
        query_vector = embedder.encode(req.query).tolist()

        # 2. Query Supabase vector match RPC function
        rpc_params = {
            "query_embedding": query_vector,
            "match_threshold": 0.3,
            "match_count": req.top_k,
            "filter_grade": req.grade,
            "filter_subject": req.subject,
        }

        response = supabase.rpc("match_textbook_chunks", rpc_params).execute()
        chunks = response.data or []

        # Fallback search if strict grade/subject filter yields no matches
        if not chunks:
            response = supabase.rpc(
                "match_textbook_chunks",
                {
                    "query_embedding": query_vector,
                    "match_threshold": 0.1,
                    "match_count": req.top_k,
                    "filter_grade": None,
                    "filter_subject": None,
                },
            ).execute()
            chunks = response.data or []

        # 3. Format context & links dynamically
        context_blocks = []
        sources = []

        for idx, item in enumerate(chunks, 1):
            text_content = item.get("chunk_text") or item.get("content") or ""
            file_url = item.get("file_url", "")
            page_num = item.get("page_number", 1)

            full_pdf_link = f"{file_url}#page={page_num}" if file_url else ""

            context_blocks.append(
                f"[Source {idx} - Grade {item.get('grade', 'N/A')} {item.get('subject', 'General')}, Page {page_num}]:\n{text_content}"
            )

            sources.append(
                {
                    "grade": item.get("grade"),
                    "subject": item.get("subject"),
                    "page_number": page_num,
                    "file_url": file_url,
                    "pdf_link": full_pdf_link,
                    "snippet": (
                        text_content[:250] + "..."
                        if len(text_content) > 250
                        else text_content
                    ),
                }
            )

        full_context = (
            "\n\n".join(context_blocks)
            if context_blocks
            else "No specific textbook context retrieved."
        )

        # 4. Prompt construction
        system_prompt = f"""You are an expert AI tutor for Ethiopian Secondary School students (Grades 9 to 12).
Answer the student's question accurately using the provided Ethiopian textbook context below.

Rules:
1. Explain concepts clearly with pedagogical warmth, structure, and high educational value.
2. If mathematical or scientific equations are involved, format them using clean inline ($...$) or display ($$...$$) LaTeX.
3. Cite your sources directly inline using brackets like [Grade X Subject, p. Y] whenever referring to factual details.
4. If the retrieved context doesn't contain enough details, answer based on general educational knowledge while explicitly stating that the exact textbook snippet wasn't retrieved.

Retrieved Textbook Context:
{full_context}

Student Question: {req.query}
"""

        # 5. Execute generation with target model
        target_model = resolve_gemini_model()
        model = genai.GenerativeModel(target_model)
        ai_response = model.generate_content(system_prompt)

        return {
            "answer": ai_response.text,
            "sources": sources,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))