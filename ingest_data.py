import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from typing import List

# --- CONFIGURATION ---
load_dotenv()
SOURCE_DATA_FILE = "data.txt"
PINECONE_INDEX_NAME = "physical-therapy-index"
EMBEDDING_MODEL = "models/text-embedding-004"

def chunk_text(text: str) -> List[str]:
    """Splits text into paragraphs for more focused embeddings."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

def main():
    print(f"--- Starting Ingestion Process for '{PINECONE_INDEX_NAME}' ---")

    # 1. Initialize Services
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not all([pinecone_api_key, gemini_api_key]):
            print("❌ ERROR: API keys not found in .env file. Exiting.")
            return

        print("Connecting to services...")
        pc = Pinecone(api_key=pinecone_api_key)
        genai.configure(api_key=gemini_api_key)
        
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"❌ ERROR: Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Exiting.")
            return
        index = pc.Index(PINECONE_INDEX_NAME)
        print("✅ Services connected successfully.")

    except Exception as e:
        print(f"❌ ERROR: Failed to initialize services: {e}")
        return

    # 2. Load and Chunk Data
    print(f"Reading and chunking data from '{SOURCE_DATA_FILE}'...")
    try:
        with open(SOURCE_DATA_FILE, 'r', encoding='utf-8') as f:
            text_data = f.read()
        text_chunks = chunk_text(text_data)
        print(f"✅ Found {len(text_chunks)} text chunks.")
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{SOURCE_DATA_FILE}' was not found. Exiting.")
        return
    
    # 3. Clear and Upload to Pinecone
    print("Clearing all old data from the index...")
    index.delete(delete_all=True)
    print("✅ Index cleared.")

    print("Generating embeddings and uploading new chunks...")
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch_chunks = text_chunks[i:i + batch_size]
        # --- This whole try/except block must be copied correctly ---
        try:
            print(f"  - Processing batch {i//batch_size + 1}...")
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_chunks,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = response['embedding']

            vectors_to_upsert = [
                {"id": f"chunk_{i+j}", "values": emb, "metadata": {"text": chunk}}
                for j, (chunk, emb) in enumerate(zip(batch_chunks, embeddings))
            ]
            index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"    ⚠️ Failed to process batch {i//batch_size + 1}. Error: {e}")
            continue # This allows the script to continue even if one batch fails
        # -----------------------------------------------------------

    print(f"✅ Ingestion complete. Check Pinecone for updated vector count.")

if __name__ == "__main__":
    main()