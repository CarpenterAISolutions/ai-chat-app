# ingest_data.py
import os
import google.generativeai as genai
from pinecone import Pinecone
import getpass

PINECONE_INDEX_NAME = "physical-therapy-index"
YOUR_TEXT_FILE = "placeholder_data.txt"

def main():
    print("--- Data Ingestion Script ---")
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY") or getpass.getpass("Please enter your Google AI (Gemini) API Key: ")
        pinecone_api_key = os.getenv("PINECONE_API_KEY") or getpass.getpass("Please enter your Pinecone API Key: ")
    except Exception as e:
        print(f"Could not read API key: {e}")
        return

    genai.configure(api_key=gemini_api_key)
    pc = Pinecone(api_key=pinecone_api_key)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Error: Index '{PINECONE_INDEX_NAME}' does not exist in Pinecone.")
        print("Please create the index in your Pinecone dashboard with dimension 768 and metric 'cosine'.")
        return

    index = pc.Index(PINECONE_INDEX_NAME)

    try:
        with open(YOUR_TEXT_FILE, 'r', encoding='utf-8') as f:
            text_data = f.read()
        chunks = [chunk.strip() for chunk in text_data.split('\n\n') if chunk.strip()]
        print(f"Found {len(chunks)} text chunks to process from '{YOUR_TEXT_FILE}'.")
    except FileNotFoundError:
        print(f"Error: The file '{YOUR_TEXT_FILE}' was not found.")
        return

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=batch_chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = response['embedding']

        vectors_to_upsert = []
        for j, chunk in enumerate(batch_chunks):
            vector_id = f"chunk_{i+j}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embeddings[j],
                "metadata": {"text": chunk}
            })

        index.upsert(vectors=vectors_to_upsert)

    print("\nData ingestion complete!")
    print(f"Total vectors in index: {index.describe_index_stats()['total_vector_count']}")

if __name__ == "__main__":
    main()
