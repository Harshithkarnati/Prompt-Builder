from sentence_transformers import SentenceTransformer
import faiss, numpy as np
import json

# Load prompts
with open("C:/Users/harsh/Desktop/Ai project/Prompt-Builder/data/processed_prompts.json") as f:
    prompts = json.load(f)

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(prompts, show_progress_bar=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Retrieval function
def retrieve_prompts(query, top_k=3):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), top_k)
    return [prompts[i] for i in I[0]]
