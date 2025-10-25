# 🚀 Prompt-Builder

A RAG-based (Retrieval-Augmented Generation) system for intelligent prompt template retrieval and matching. This project uses vector embeddings to find the most relevant prompt templates based on vague user queries, with dynamic weightage scoring.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

Prompt-Builder is a template retrieval system that:
1. **Indexes** prompt templates into a vector database
2. **Retrieves** the top 3 most relevant templates for any vague user prompt
3. **Scores** each template with dynamic weightage based on similarity (45%, 35%, 20% pattern)
4. **Sorts** results by relevance in descending order

## ✨ Features

- 🔍 **Semantic Search**: Uses sentence transformers for intelligent matching
- 📊 **Dynamic Scoring**: Automatic weightage calculation for retrieved templates
- 💾 **Vector Database**: ChromaDB for persistent storage and fast retrieval
- 🎯 **Top-K Retrieval**: Gets the 3 most relevant templates per query
- 📈 **Progress Tracking**: Built-in progress bars for batch operations
- 🔄 **JSONL Output**: Structured output for easy downstream processing

## 📁 Project Structure

```
Prompt-Builder/
│
├── .env                          # Environment variables (YOU CREATE THIS)
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── data/
│   ├── templates/
│   │   └── templates.json        # Your prompt templates (required)
│   ├── generated_vague/
│   │   └── generated_vague_prompts.json  # Vague prompts for testing
│   ├── retrieval/
│   │   └── pure_python_results.jsonl    # Retrieved templates with metadata
│   └── training/
│       ├── t5_fine_tuning_dataset.jsonl    # Basic T5 training data
│       └── enhanced_t5_dataset.jsonl       # Enhanced T5 training data ⭐
│
├── my_vector_db/                 # ChromaDB storage (auto-created)
│   └── chroma.sqlite3
│
├── scripts/                      # T5 Dataset Building Scripts ✨
│   ├── build_t5_dataset.py      # Basic T5 dataset builder
│   └── build_enhanced_t5_dataset.py  # Enhanced T5 dataset builder ⭐
│
└── training_model/
    ├── indexing_templates.py     # Step 1: Index templates into vector DB
    └── retrieve_templates.py     # Step 2: Retrieve templates for vague prompts
```

## 🔧 Prerequisites

- Python 3.8 or higher
- Windows (PowerShell) / Linux / macOS
- At least 2GB RAM (for embedding models)
- Internet connection (first run only, for downloading models)

## 📦 Installation

### Step 1: Clone the Repository

```powershell
git clone https://github.com/Harshithkarnati/Prompt-Builder.git
cd Prompt-Builder
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv env

# Activate it
.\env\Scripts\Activate.ps1
```

> **Note**: If you get an execution policy error on Windows, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3: Install Dependencies



**Or** install from requirements.txt (includes extra packages):
```powershell
pip install -r requirements.txt
```

### Step 4: Verify Installation

```powershell
python -c "import chromadb, sentence_transformers, tqdm, dotenv; print('✅ All packages installed successfully!')"
```

## ⚙️ Environment Setup

### Create `.env` File

Create a file named `.env` in the project root directory:

```powershell
# Windows PowerShell
New-Item -Path .env -ItemType File
```

### Configure `.env` File

Open `.env` and add the following configuration:

```env
# Path to your templates JSON file
TEMPLATES_FILE=C:...../Prompt-Builder/data/templates/templates.json

# Path to vague prompts file (for retrieval)
VAGUE_PROMPTS_FILE=C:....../Ai project/Prompt-Builder/data/generated_vague/generated_vague_prompts.json

# Output file for retrieved templates
INTERMEDIATE_FILE=C:...../Ai project/Prompt-Builder/data/intermediate_data.jsonl

# Vector database directory
MY_VECTOR_DB_DIR=./my_vector_db
```

> **⚠️ Important Notes:**
> - Use **forward slashes** (`/`) or **double backslashes** (`\\`) in paths
> - Do **NOT** wrap paths in quotes
> - Adjust paths to match your actual project location
> - All paths should be absolute except `MY_VECTOR_DB_DIR` (relative is fine)

### Template JSON Format

Your `data/templates/templates.json` should follow this structure:

```json
[
  {
    "id": "template_001",
    "template_name": "Marketing Campaign Brief",
    "template_description": "A comprehensive template for planning marketing campaigns",
    "category": "Marketing",
    "template_format": "Campaign Name: [NAME]\nObjective: [OBJECTIVE]\nTarget Audience: [AUDIENCE]\nKey Messages: [MESSAGES]"
  },
  {
    "id": "template_002",
    "template_name": "Code Review Checklist",
    "template_description": "A template for conducting thorough code reviews",
    "category": "Development",
    "template_format": "Code Quality:\n- [ ] Readable\n- [ ] Documented\n- [ ] Tested"
  }
]
```

### Vague Prompts JSON Format

Your `data/generated_vague/generated_vague_prompts.json` should look like:

```json
[
  {
    "user_vague_prompt": "I need help planning a marketing strategy",
    "context": "Business planning"
  },
  {
    "user_vague_prompt": "How do I review code effectively?",
    "context": "Software development"
  }
]
```

## 🚀 Usage

### Step 1: Index Templates (Run Once)

This creates embeddings and stores them in the vector database:

```powershell
cd training_model
python indexing_templates.py
```

**Expected Output:**
```
Loading embedding model...
Loading templates from JSON...
Preparing 500 templates for embedding...
Embedding documents and adding to ChromaDB...
✅ RAG Indexing Complete! Your vector DB is ready.
```

### Step 2: Retrieve Templates

This finds the top 3 templates for each vague prompt:

```powershell
python retrieve_templates.py
```

**Expected Output:**
```
--- Starting Part 1: Template Retrieval ---
Loading embedding model: all-MiniLM-L6-v2
Connected to existing collection: 'prompt_templates'
Loading vague prompts from ...
Starting retrieval for 100 prompts...
Retrieving Templates: 100%|████████| 100/100 [00:15<00:00, 6.67it/s]

--- Template Retrieval Complete ---
✅ Successfully retrieved templates for 100 prompts.
Intermediate data saved to: ../data/intermediate_data.jsonl
```

### Step 3: Build T5 Fine-tuning Dataset ✨

Transform the retrieved templates into optimized prompts for T5 training:

#### Option A: Basic Version
```powershell
cd scripts
python build_t5_dataset.py
```

#### Option B: Enhanced Version (Recommended) ⭐
```powershell
cd scripts
python build_enhanced_t5_dataset.py
```

**Expected Output:**
```
Building enhanced T5 dataset...
Input: data/retrieval/pure_python_results.jsonl
Output: data/training/enhanced_t5_dataset.jsonl
Processed 100 entries...
...
Enhanced dataset building complete!
Successfully processed: 958 entries
Output saved to: data/training/enhanced_t5_dataset.jsonl
```

The enhanced version creates intelligent, context-aware prompts with:
- 🎯 Content type detection (social media, business, presentations)
- 👤 User personalization based on metadata
- 📝 Professional formatting ready for AI consumption
- 🔧 Specialized templates for different domains

### Output Format

The `intermediate_data.jsonl` file contains:

```json
{
  "vague_prompt_data": {
    "user_vague_prompt": "I need help planning a marketing strategy",
    "context": "Business planning"
  },
  "retrieved_templates": [
    {
      "template_name": "Marketing Campaign Brief",
      "template_format": "Campaign Name: [NAME]...",
      "weightage_percent": 47
    },
    {
      "template_name": "Strategic Planning Template",
      "template_format": "Vision: [VISION]...",
      "weightage_percent": 34
    },
    {
      "template_name": "Business Proposal",
      "template_format": "Executive Summary: [SUMMARY]...",
      "weightage_percent": 19
    }
  ]
}
```

## 🔍 How It Works

### 1. **Indexing Phase** (`indexing_templates.py`)

```
Templates JSON → Embedding Model → Vector Database
```

- Loads all templates from `templates.json`
- Creates semantic embeddings using `all-MiniLM-L6-v2`
- Stores embeddings + metadata in ChromaDB
- Metadata includes: template_name, template_format, category

### 2. **Retrieval Phase** (`retrieve_templates.py`)

```
Vague Prompt → Embedding → Vector Search → Top 3 Templates + Weightage
```

- Embeds each vague prompt
- Performs similarity search in vector DB
- Retrieves top 3 most similar templates
- Calculates dynamic weightage based on distance scores
- Sorts by weightage (highest first)
- Outputs to JSONL file

### 3. **Weightage Calculation**

The system uses similarity scores to distribute 100% across 3 templates:
- **Formula**: `similarity = 1 - distance`
- **Normalization**: Each similarity / total * 100
- **Pattern**: Typically results in ~45%, ~35%, ~20% distribution
- **Sorting**: Results sorted by weightage in descending order

## 🐛 Troubleshooting

### Error: `OSError: [Errno 22] Invalid argument`

**Cause**: Path in `.env` has quotes or invalid backslashes

**Solution**: 
```env
# ❌ Wrong
TEMPLATES_FILE="C:\Users\...\templates.json"

# ✅ Correct
TEMPLATES_FILE=C:/Users/.../templates.json
```

### Error: `NameError: name 'TEMPLATES_FILE' is not defined`

**Cause**: Environment variables not loaded

**Solution**: 
1. Check `.env` file exists in project root
2. Ensure `load_dotenv()` is called in script
3. Verify paths don't have quotes

### Error: `Collection is empty`

**Cause**: Templates not indexed yet

**Solution**: Run `python indexing_templates.py` first

### Error: `File not found: templates.json`

**Cause**: Wrong path in `.env`

**Solution**: Use absolute paths and verify file exists:
```powershell
Test-Path "C:/Users/harsh/Desktop/Ai project/Prompt-Builder/data/templates/templates.json"
```

### Model Download Issues

**Cause**: No internet connection on first run

**Solution**: Ensure internet connection when running for the first time (downloads ~80MB model)

### Memory Issues

**Cause**: Large number of templates or insufficient RAM

**Solution**: 
- Process templates in batches
- Use a machine with at least 2GB RAM
- Close other applications

## 📝 Notes

- First run downloads the `all-MiniLM-L6-v2` model (~80MB)
- Vector database is persistent (no need to re-index unless templates change)
- ChromaDB creates a SQLite database in `my_vector_db/`
- Intermediate data is in JSONL format (one JSON object per line)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Harshith Karnati**
- GitHub: [@Harshithkarnati](https://github.com/Harshithkarnati)

---

**Happy Prompting! 🎉**
