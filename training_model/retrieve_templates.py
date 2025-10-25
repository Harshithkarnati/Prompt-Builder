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
    Calculate dynamic weightage based on similarity scores with robust error handling.
    Converts distances to percentages ensuring values are always valid (0-100).
    """
    # Handle edge cases
    if not distances or len(distances) == 0:
        return []
    
    # Log unusual distance values for debugging
    if any(d > 1 for d in distances):
        print(f"  [WARNING] Found distances > 1: {[round(d, 4) for d in distances]}")
    if any(d < 0 for d in distances):
        print(f"  [WARNING] Found negative distances: {[round(d, 4) for d in distances]}")
    
    # Handle fewer than 3 results
    if len(distances) == 1:
        return [100]
    elif len(distances) == 2:
        # For 2 results, use 60-40 split based on relative similarity
        distances = [max(0.0, min(1.0, float(d))) for d in distances]
        similarities = [1.0 - d for d in distances]
        if sum(similarities) <= 0.001:
            return [50, 50]
        total = sum(similarities)
        weights = [round((sim / total) * 100) for sim in similarities]
        # Ensure sum equals 100
        if sum(weights) != 100:
            weights[0] = 100 - weights[1]
        return [max(1, w) for w in weights]
    
    # Clamp distances to valid [0, 1] range
    distances = [max(0.0, min(1.0, float(d))) for d in distances]
    
    # Convert to similarities (higher = better)
    similarities = [1.0 - d for d in distances]
    
    # Handle edge case: all similarities are essentially zero
    total = sum(similarities)
    if total <= 0.001:
        # Use default pattern for no similarity
        return [45, 35, 20] if len(distances) >= 3 else [50, 50]
    
    # Handle edge case: all similarities are nearly equal
    if len(set([round(s, 3) for s in similarities])) == 1:
        # Use default graduated pattern for equal similarities
        if len(distances) >= 3:
            return [45, 35, 20]
        else:
            return [50, 50]
    
    # Calculate percentages
    percentages = [(sim / total) * 100 for sim in similarities]
    
    # Round to integers with minimum 1%
    percentages = [max(1, round(p)) for p in percentages]
    
    # Adjust to ensure sum equals exactly 100
    current_sum = sum(percentages)
    diff = 100 - current_sum
    
    if diff != 0:
        # Distribute difference starting with highest percentage
        sorted_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)
        abs_diff = abs(diff)
        direction = 1 if diff > 0 else -1
        
        for i in range(abs_diff):
            idx = sorted_indices[i % len(sorted_indices)]
            if direction > 0 or percentages[idx] > 1:  # Don't let any go below 1%
                percentages[idx] += direction
    
    # Final safety check
    percentages = [max(1, min(98, p)) for p in percentages]
    
    # One more adjustment to ensure exactly 100
    final_sum = sum(percentages)
    if final_sum != 100:
        percentages[0] += (100 - final_sum)
        percentages[0] = max(1, percentages[0])
    
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
                
                # Validate distances
                if not distances or len(distances) == 0:
                    print(f"  [ERROR] No results returned for: '{vague_prompt_text[:50]}...'. Skipping.")
                    continue
                
                # Extract the metadata (which contains our template_format)
                retrieved_templates_metadata = results['metadatas'][0]
                
                # We also want the names for our meta-prompt
                retrieved_names = [meta.get('template_name', 'Unknown') for meta in retrieved_templates_metadata]
                retrieved_formats = [meta.get('template_format', '') for meta in retrieved_templates_metadata]
                
                # Validate that we have the same number of items
                if len(distances) != len(retrieved_names) or len(distances) != len(retrieved_formats):
                    print(f"  [ERROR] Mismatched result lengths for: '{vague_prompt_text[:50]}...'. Skipping.")
                    continue
                
                # Calculate weightage for the retrieved templates
                weightages = calculate_weightage(distances)
                
                # Validate weightages
                if not weightages or len(weightages) != len(distances):
                    print(f"  [ERROR] Invalid weightages calculated for: '{vague_prompt_text[:50]}...'. Skipping.")
                    continue
                
                # Final validation: ensure weightages are valid
                if any(w < 0 or w > 100 for w in weightages):
                    print(f"  [ERROR] Invalid weightage values {weightages} for: '{vague_prompt_text[:50]}...'. Skipping.")
                    continue
                
                if abs(sum(weightages) - 100) > 1:  # Allow 1% tolerance for rounding
                    print(f"  [WARNING] Weightages don't sum to 100: {weightages} (sum={sum(weightages)}) for: '{vague_prompt_text[:50]}...'")

                # Combine names, formats, and weightages
                retrieved_templates = [
                                        {
                                            "template_name": name, 
                                            "template_format": fmt,
                                            "weightage_percent": weight,
                                            "similarity_distance": round(dist, 4)  # Keep original distance for debugging
                                        }
                                        for name, fmt, weight, dist in zip(retrieved_names, retrieved_formats, weightages, distances)
                        ]
                retrieved_templates.sort(key=lambda x: x['weightage_percent'], reverse=True)
            except Exception as e:
                print(f"  [ERROR] RAG query failed for: '{vague_prompt_text[:50]}...'. Skipping. Error: {e}")
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