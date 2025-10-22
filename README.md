# 🚀 Prompt-Builder

An AI-powered prompt optimization system that uses semantic search and T5 language models to improve and enhance user prompts.

## 📋 Features

- **Semantic Search**: Retrieves similar prompts using sentence transformers and FAISS
- **T5 Optimization**: Optimizes prompts using Google's FLAN-T5 model
- **REST API**: FastAPI-based backend with automatic documentation
- **Web Interface**: Simple, beautiful frontend for easy interaction
- **Prompt Variations**: Generate multiple variations of optimized prompts
- **Modular Architecture**: Clean separation of concerns with reusable components

## 🏗️ Architecture

```
Prompt-Builder/
├── api/                    # FastAPI REST API
│   └── main.py            # API endpoints and server
├── models/                 # AI Models
│   └── t5_prompt_optimizer/
│       └── t5.py          # T5 model wrapper
├── utils/                  # Utility functions
│   ├── preprocessing.py   # Prompt preprocessing
│   └── retrieval.py       # Semantic search & retrieval
├── embeddings/            # Embedding utilities
│   └── embedding.py       # Sentence transformers wrapper
├── data/                  # Data storage
│   ├── raw_prompts.json   # Raw prompt data
│   └── processed_prompts.json  # Processed prompts
├── frontend/              # Web interface
│   └── index.html        # Single-page application
├── main.py               # CLI entry point
└── requirements.txt      # Python dependencies
```

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Harshithkarnati/Prompt-Builder.git
   cd Prompt-Builder
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   ```

3. **Activate virtual environment**
   - Windows:
     ```powershell
     .\env\Scripts\Activate.ps1
     ```
   - Linux/Mac:
     ```bash
     source env/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Start the API Server

```bash
python main.py api
```

The API will start on `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Open the Web Interface

Simply open `frontend/index.html` in your web browser, or use Python's HTTP server:

```bash
cd frontend
python -m http.server 8080
```

Then visit: `http://localhost:8080`

### Test from Command Line

```bash
python main.py test "Write a function to sort an array"
```

## 📡 API Endpoints

### POST `/generate_prompt`
Generate an optimized prompt with semantic search context.

**Request:**
```json
{
  "user_prompt": "write code to sort array",
  "top_k": 3,
  "include_variations": false
}
```

**Response:**
```json
{
  "original_prompt": "write code to sort array",
  "optimized_prompt": "Write a Python function that implements an efficient sorting algorithm...",
  "retrieved_prompts": ["...", "...", "..."],
  "variations": null
}
```

### GET `/health`
Check API health status.

### POST `/retrieve`
Retrieve similar prompts only (no optimization).

### POST `/optimize`
Optimize prompt without retrieval context.

## 🔬 How It Works

1. **User Input**: User provides a prompt through web interface or API
2. **Semantic Search**: System encodes prompt and retrieves similar prompts from database using FAISS
3. **T5 Optimization**: Retrieved prompts provide context for T5 model to optimize the original prompt
4. **Response**: Returns optimized prompt with context and optional variations

## 🛠️ Development

### Project Structure

- **api/main.py**: FastAPI application with CORS, endpoints, and model initialization
- **models/t5_prompt_optimizer/t5.py**: T5 model wrapper with optimization logic
- **utils/retrieval.py**: FAISS-based semantic search implementation
- **utils/preprocessing.py**: Text cleaning and preprocessing utilities
- **frontend/index.html**: Responsive web UI with real-time API communication

### Adding New Prompts

Add prompts to `data/processed_prompts.json`:

```json
[
  {
    "prompt": "Your prompt here",
    "category": "category_name",
    "model_target": "T5"
  }
]
```

Then restart the API to rebuild the FAISS index.

### Customizing Models

Edit `models/t5_prompt_optimizer/t5.py` to change:
- Model size (e.g., `google/flan-t5-base`, `google/flan-t5-large`)
- Generation parameters (temperature, max_length, etc.)
- Prompt instruction templates

## 📦 Dependencies

Key libraries:
- **FastAPI**: Modern web framework for APIs
- **sentence-transformers**: Semantic embeddings
- **transformers**: Hugging Face T5 models
- **faiss-cpu**: Efficient similarity search
- **uvicorn**: ASGI server
- **torch**: PyTorch for model inference

See `requirements.txt` for complete list.

## 🎯 Use Cases

- **Content Creation**: Improve writing prompts for AI models
- **Code Generation**: Optimize programming task descriptions
- **Educational**: Enhance learning prompts and questions
- **Research**: Refine research questions and hypotheses
- **Business**: Create better prompts for AI assistants

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Harshith Karnati**
- GitHub: [@Harshithkarnati](https://github.com/Harshithkarnati)

## 🙏 Acknowledgments

- Google FLAN-T5 team for the language model
- Sentence Transformers for semantic embeddings
- FastAPI for the excellent framework
- FAISS for efficient similarity search 
