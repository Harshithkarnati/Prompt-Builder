import json
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
TEMPLATES_FILE = os.getenv('TEMPLATES_FILE')
MY_VECTOR_DB_DIR = os.getenv('MY_VECTOR_DB_DIR')

# 1. Load your embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Setup local vector database
client = chromadb.PersistentClient(path=MY_VECTOR_DB_DIR) # This creates a folder
collection = client.get_or_create_collection(name="prompt_templates")

# 3. Load your 500 templates
print("Loading templates from JSON...")
with open(TEMPLATES_FILE, 'r', encoding='utf-8') as f:
    templates = json.load(f) # Assumes a list of templates in this file
    
documents_to_embed = []
metadata_list = []
ids_list = []

# 4. Prepare data for embedding
print(f"Preparing {len(templates)} templates for embedding...")
for template in templates:
    # THIS IS THE KEY: Create a clean, descriptive text for embedding
    text_to_embed = f"Name: {template['template_name']}\n" \
                    f"Description: {template['template_description']}\n" \
                    f"Category: {template['category']}"
    
    documents_to_embed.append(text_to_embed)
    
    # Store the *actual* useful data (the format) as metadata
    # This is what you want to get back *after* the search
    metadata_list.append({
        "template_name": template['template_name'],
        "template_format": template['template_format'],
        "category": template['category']
    })
    
    # Give each item a unique ID
    ids_list.append(template['id'])

# 5. Embed and store in the database
print("Embedding documents and adding to ChromaDB...")
# ChromaDB can handle the embedding for you if you pass it the model
# But doing it in batches with sentence-transformers is often faster
# For simplicity, let's generate embeddings first:
embeddings = model.encode(documents_to_embed, show_progress_bar=True)

collection.add(
    embeddings=embeddings,
    metadatas=metadata_list,
    ids=ids_list
)

print("✅ RAG Indexing Complete! Your vector DB is ready.")