from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from typing import List, Tuple

class PromptRetriever:
    """Retrieves similar prompts using semantic search with FAISS."""
    
    def __init__(self, prompts_path: str = "data/processed_prompts.json", 
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the retriever with prompts and embedding model.
        
        Args:
            prompts_path: Path to processed prompts JSON file
            model_name: Name of sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.prompts = self._load_prompts(prompts_path)
        self.index = None
        self._build_index()
    
    def _load_prompts(self, path: str) -> List[str]:
        """Load prompts from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list of strings and list of dicts
            if data and isinstance(data[0], dict):
                return [item['prompt'] for item in data if 'prompt' in item]
            return data
        except Exception as e:
            print(f"Error loading prompts: {e}")
            # Return default prompts if file not found
            return [
                "Write a short story about a robot learning emotions.",
                "Summarize the following article in three sentences.",
                "Translate the following English paragraph to French.",
                "Explain quantum physics concepts to a 12-year-old.",
                "Generate a Python function that reverses a string.",
                "Create a haiku about the changing seasons."
            ]
    
    def _build_index(self):
        """Build FAISS index from prompt embeddings."""
        print(f"Building FAISS index for {len(self.prompts)} prompts...")
        embeddings = self.model.encode(self.prompts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"FAISS index built successfully with dimension {dimension}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve most similar prompts to the query.
        
        Args:
            query: User's input prompt
            top_k: Number of similar prompts to retrieve
            
        Returns:
            List of (prompt, distance) tuples
        """
        if not self.index:
            raise RuntimeError("FAISS index not built")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(top_k, len(self.prompts))
        )
        
        # Return prompts with their distances
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.prompts):
                results.append((self.prompts[idx], float(dist)))
        
        return results

# Global retriever instance
_retriever = None

def get_retriever() -> PromptRetriever:
    """Get or create global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = PromptRetriever()
    return _retriever

def retrieve_prompts(query: str, top_k: int = 3) -> List[str]:
    """
    Convenience function to retrieve similar prompts.
    
    Args:
        query: User's input prompt
        top_k: Number of similar prompts to retrieve
        
    Returns:
        List of similar prompt strings
    """
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k)
    return [prompt for prompt, _ in results]

if __name__ == "__main__":
    # Test retrieval
    test_query = "Write code to sort an array"
    print(f"\nQuery: {test_query}")
    print("\nRetrieved prompts:")
    for i, prompt in enumerate(retrieve_prompts(test_query, top_k=3), 1):
        print(f"{i}. {prompt}")
