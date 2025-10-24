import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import os 

# Load environment variables from .env file
load_dotenv()



# --- CONFIGURATION ---
VAGUE_PROMPTS_FILE = os.getenv('VAGUE_PROMPTS_FILE')
INTERMEDIATE_FILE = os.getenv('INTERMEDIATE_FILE')
MY_VECTOR_DB_DIR = os.getenv('MY_VECTOR_DB_DIR')

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DB_PATH = MY_VECTOR_DB_DIR
COLLECTION_NAME = "prompt_templates"
# ---------------------


def calculate_weightage(distances):
    """
    Calculate dynamic weightage based on similarity scores.
    Converts distances to percentages with pattern: ~45%, ~35%, ~20%
    """
    # Convert distances to similarity scores (smaller distance = higher similarity)
    # For cosine distance: similarity = 1 - distance
    similarities = [1 - d for d in distances]
    
    # Normalize to sum to 100
    total = sum(similarities)
    if total == 0:
        # Fallback if all distances are 1 (no similarity)
        return [45, 35, 20]
    
    # Calculate percentages
    percentages = [(sim / total) * 100 for sim in similarities]
    
    # Round to integers and adjust to ensure they sum to 100
    percentages = [round(p) for p in percentages]
    diff = 100 - sum(percentages)
    if diff != 0:
        percentages[0] += diff  # Adjust the first (highest) percentage
    
    return percentages

def load_vague_prompts(filepath):
    """Loads the vague prompts data from a JSON file."""
    print(f"Loading vague prompts from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return []

def main():
    print("--- Starting Part 1: Template Retrieval ---")
    
    # 1. Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 2. Connect to vector DB
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Connected to existing collection: '{COLLECTION_NAME}'")
        if collection.count() == 0:
            print("Error: The collection is empty. Run '01_index_templates.py' first.")
            return
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        print("Have you run '01_index_templates.py' yet?")
        return

    # 3. Load vague prompts
    vague_prompts = load_vague_prompts(VAGUE_PROMPTS_FILE)
    if not vague_prompts:
        print("No vague prompts found. Exiting.")
        return

    # 4. Open intermediate output file
    retrieved_count = 0
    with open(INTERMEDIATE_FILE, 'w', encoding='utf-8') as f:
        print(f"Opened intermediate file: {INTERMEDIATE_FILE}")
        print(f"Starting retrieval for {len(vague_prompts)} prompts...")
        
        # 5. Start the retrieval loop
        for prompt_data in tqdm(vague_prompts, desc="Retrieving Templates"):
            vague_prompt_text = prompt_data['user_vague_prompt']
            
            # 6. Embed the vague prompt
            try:
                query_embedding = model.encode(vague_prompt_text).tolist()
            except Exception as e:
                print(f"  [ERROR] Embedding failed for: '{vague_prompt_text}'. Skipping. Error: {e}")
                continue
                
            # 7. Query RAG for top 3 templates
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3
                )
                distances = results['distances'][0] 
                # Extract the metadata (which contains our template_format)
                retrieved_templates_metadata = results['metadatas'][0]
                
                # We also want the names for our meta-prompt
                retrieved_names = [meta.get('template_name', 'Unknown') for meta in retrieved_templates_metadata]
                retrieved_formats = [meta.get('template_format', '') for meta in retrieved_templates_metadata]
                
                # Calculate weightage for the top 3
                weightages = calculate_weightage(distances)

                # Combine names, formats, and weightages
                retrieved_templates = [
                                        {
                                            "template_name": name, 
                                            "template_format": fmt,
                                            "weightage_percent": weight
                                        }
                                        for name, fmt, weight in zip(retrieved_names, retrieved_formats, weightages)
                        ]
                retrieved_templates.sort(key=lambda x: x['weightage_percent'], reverse=True)
            except Exception as e:
                print(f"  [ERROR] RAG query failed for: '{vague_prompt_text}'. Skipping. Error: {e}")
                continue
                
            # 8. Create the intermediate data object
            intermediate_data = {
                "vague_prompt_data": prompt_data,
                "retrieved_templates": retrieved_templates
            }
            
            # 9. Write to the JSONL file
            f.write(json.dumps(intermediate_data) + "\n")
            retrieved_count += 1
            
    print("\n--- Template Retrieval Complete ---")
    print(f"✅ Successfully retrieved templates for {retrieved_count} prompts.")
    print(f"Intermediate data saved to: {INTERMEDIATE_FILE}")

if __name__ == "__main__":
    main()