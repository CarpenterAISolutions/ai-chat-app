import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()
PINECONE_INDEX_NAME = "physical-therapy-index"
EMBEDDING_MODEL = "models/text-embedding-004"
TEST_QUERY = "what is the RICE method"

def main():
    print("--- Running Pinecone Verification Test ---")
    try:
        # Initialize services
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
        print("✅ Services connected successfully.")

        # Search Pinecone
        print(f"Searching for test query: '{TEST_QUERY}'")
        query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=TEST_QUERY, task_type="RETRIEVAL_QUERY")["embedding"]
        search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        # Print results
        print("\n--- PINEcone Search Results ---")
        if search_results['matches']:
            for match in search_results['matches']:
                print(f"  - Score: {match['score']:.4f}")
                print(f"    Text: \"{match['metadata']['text'][:100]}...\"")
            print("\n✅ TEST PASSED: Successfully retrieved data from Pinecone.")
        else:
            print("\n❌ TEST FAILED: Pinecone search returned no matches.")

    except Exception as e:
        print(f"❌ An unexpected error occurred during the test: {e}")

if __name__ == "__main__":
    main()