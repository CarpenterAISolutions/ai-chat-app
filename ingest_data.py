import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from typing import List

load_dotenv()
SOURCE_DATA_FILE = "data.txt"
PINECONE_INDEX_NAME = "physical-therapy-index"
EMBEDDING_MODEL = "models/text-embedding-004"

def chunk_text(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

def main():
    print(f"--- Starting Ingestion for index '{PINECONE_INDEX_NAME}' ---")
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not all([pinecone_api_key, gemini_api_key]):
            raise ValueError("API keys not found in .env file.")

        print("Connecting to services...")
        pc = Pinecone(api_key=pinecone_api_key)
        genai.configure(api_key=gemini_api_key)
        
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")
        index = pc.Index(PINECONE_INDEX_NAME)
        print("✅ Services connected.")

        print(f"Reading and chunking data from '{SOURCE_DATA_FILE}'...")
        with open(SOURCE_DATA_FILE, 'r', encoding='utf-8') as f:
            text_data = f.read()
        text_chunks = chunk_text(text_data)
        print(f"✅ Found {len(text_chunks)} text chunks.")

        print("Clearing all old data from the index...")
        index.delete(delete_all=True)
        print("✅ Index cleared.")

        print("Uploading new chunks...")
        batch_size = 100
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            print(f"  - Processing batch {i//batch_size + 1}...")
            response = genai.embed_content(model=EMBEDDING_MODEL, content=batch, task_type="RETRIEVAL_DOCUMENT")
            embeddings = response['embedding']
            vectors = [{"id": f"chunk_{i+j}", "values": emb, "metadata": {"text": chunk}} for j, (chunk, emb) in enumerate(zip(batch, embeddings))]
            index.upsert(vectors=vectors)
        
        print(f"✅ Ingestion complete. Final vector count: {index.describe_index_stats()['total_vector_count']}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()