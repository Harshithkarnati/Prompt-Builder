import json
import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.,!?-]', '', text)
    return text.strip()

def preprocess_prompts(raw_prompts: List[Dict]) -> List[str]:
    """
    Preprocess raw prompts for embedding and retrieval.
    
    Args:
        raw_prompts: List of prompt dictionaries with 'prompt' field
        
    Returns:
        List of cleaned prompt strings
    """
    processed = []
    for item in raw_prompts:
        if isinstance(item, dict) and 'prompt' in item:
            cleaned = clean_text(item['prompt'])
            processed.append(cleaned)
        elif isinstance(item, str):
            cleaned = clean_text(item)
            processed.append(cleaned)
    return processed

def load_and_preprocess(input_path: str, output_path: str):
    """
    Load raw prompts, preprocess them, and save to output file.
    
    Args:
        input_path: Path to raw prompts JSON file
        output_path: Path to save processed prompts
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # If raw_data is empty or not a list, use default prompts
        if not raw_data or not isinstance(raw_data, list):
            print("No raw prompts found, using defaults from processed_prompts.json")
            return
        
        processed = preprocess_prompts(raw_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
        print(f"Preprocessed {len(processed)} prompts and saved to {output_path}")
        
    except Exception as e:
        print(f"Error preprocessing prompts: {e}")
        raise

if __name__ == "__main__":
    load_and_preprocess(
        "data/raw_prompts.json",
        "data/processed_prompts.json"
    )
