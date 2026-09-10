import os
import re
import time
import pymupdf
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# ==============================================================================
# 1. Configuration & Credentials
# ==============================================================================
# REPLACE THESE WITH YOUR ACTUAL SUPABASE CREDS IF NOT SET IN ENVIRONMENT VARIABLES
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://oqvrruureotvhpxfaeel.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9xdnJydXVyZW90dmhweGZhZWVsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4ODUwMzc4MywiZXhwIjoyMTA0MDc5NzgzfQ.S_8Ezltydu4TXUuMWiLdCIukzUgqIX9wyB7oEvbY7Is")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

TEXTBOOKS_DIR = "./textbooks"
TARGET_FILES = ["G11-Physics.pdf"]

# Retry helper that raises an exception if all retries fail
def upload_with_retry(table_name, payload, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            supabase.table(table_name).insert(payload).execute()
            return True
        except Exception as e:
            print(f"    [Upload Attempt {attempt}/{max_retries} Failed]: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise ConnectionError("Failed to upload batch to Supabase after 3 attempts. Aborting.")

# ==============================================================================
# 2. Targeted Ingestion Routine
# ==============================================================================
def ingest_selected_textbooks():
    if not os.path.exists(TEXTBOOKS_DIR):
        print(f"Directory '{TEXTBOOKS_DIR}' not found.")
        return

    print(f"Processing target files: {TARGET_FILES}\n")

    for filename in TARGET_FILES:
        pdf_path = os.path.join(TEXTBOOKS_DIR, filename)
        if not os.path.exists(pdf_path):
            print(f"File {filename} not found in {TEXTBOOKS_DIR}.")
            continue

        match = re.match(r"G(\d+)-([A-Za-z]+)\.pdf", filename, re.IGNORECASE)
        if not match:
            print(f"Skipping {filename}: Incorrect filename pattern.")
            continue

        grade = int(match.group(1))
        subject = match.group(2).capitalize()

        print(f"--> Processing Grade {grade} {subject} ({filename})...")

        doc = pymupdf.open(pdf_path)
        chunks_to_insert = []
        uploaded_count = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if len(text) < 50:
                continue

            clean_text = " ".join(text.split())
            vector = embedding_model.encode(clean_text).tolist()

            chunks_to_insert.append({
                "grade": grade,
                "subject": subject,
                "page_number": page_num + 1,
                "content": clean_text,
                "embedding": vector
            })

            # Upload in batches of 50 pages
            if len(chunks_to_insert) >= 50:
                upload_with_retry("textbook_chunks", chunks_to_insert)
                uploaded_count += len(chunks_to_insert)
                print(f"    Uploaded {uploaded_count} pages...")
                chunks_to_insert = []

        # Insert remaining pages
        if chunks_to_insert:
            upload_with_retry("textbook_chunks", chunks_to_insert)
            uploaded_count += len(chunks_to_insert)

        print(f"    SUCCESS: Indexing complete for {filename} ({uploaded_count} pages saved to Supabase).\n")

    print("Targeted ingestion complete!")

if __name__ == "__main__":
    ingest_selected_textbooks()