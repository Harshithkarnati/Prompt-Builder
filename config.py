"""
Configuration file for Prompt Builder
"""

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Model Configuration
T5_MODEL_NAME = "google/flan-t5-small"  # Options: flan-t5-small, flan-t5-base, flan-t5-large
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval Configuration
DEFAULT_TOP_K = 3
MAX_TOP_K = 10

# Generation Configuration
MAX_LENGTH = 150
TEMPERATURE = 0.7
TOP_P = 0.9

# Data Paths
RAW_PROMPTS_PATH = "data/raw_prompts.json"
PROCESSED_PROMPTS_PATH = "data/processed_prompts.json"

# FAISS Configuration
FAISS_INDEX_TYPE = "IndexFlatL2"  # L2 distance for similarity

# CORS Configuration
CORS_ORIGINS = ["*"]  # Allow all origins (adjust for production)
