# Prompt Builder - Quick Start Guide

## 🚀 Quick Start

### 1. Activate Virtual Environment
```powershell
.\env\Scripts\Activate.ps1
```

### 2. Install Dependencies (if not done)
```powershell
pip install -r requirements.txt
```

### 3. Start the API Server
```powershell
python main.py api
```

The API will start on http://localhost:8000
- API Docs: http://localhost:8000/docs
- Interactive API: http://localhost:8000/redoc

### 4. Open Web Interface
Open `frontend/index.html` in your browser, or:
```powershell
cd frontend
python -m http.server 8080
```
Then visit: http://localhost:8080

### 5. Test the System
```powershell
python test.py
```

## 📡 API Usage Examples

### Using curl (Windows PowerShell)
```powershell
$body = @{
    user_prompt = "write code to sort array"
    top_k = 3
    include_variations = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/generate_prompt" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/generate_prompt",
    json={
        "user_prompt": "write code to sort array",
        "top_k": 3,
        "include_variations": False
    }
)

result = response.json()
print("Optimized:", result["optimized_prompt"])
```

## 🛠️ CLI Commands

```powershell
# Start API with auto-reload (development)
python main.py api --reload

# Test with custom prompt
python main.py test "Your custom prompt here"

# Preprocess prompts
python main.py preprocess --input data/raw_prompts.json --output data/processed_prompts.json
```

## 📦 Project Components

✓ **API Server** (`api/main.py`)
  - FastAPI REST API
  - Auto-generated documentation
  - CORS enabled for frontend

✓ **Retrieval System** (`utils/retrieval.py`)
  - Semantic search with FAISS
  - Sentence transformer embeddings
  - Top-K similar prompt retrieval

✓ **T5 Optimizer** (`models/t5_prompt_optimizer/t5.py`)
  - FLAN-T5 model integration
  - Context-aware optimization
  - Variation generation

✓ **Preprocessing** (`utils/preprocessing.py`)
  - Text cleaning
  - Prompt normalization

✓ **Web Interface** (`frontend/index.html`)
  - Responsive design
  - Real-time API communication
  - Beautiful UI

✓ **Testing** (`test.py`)
  - Component tests
  - Integration tests
  - Full pipeline validation

## 🔧 Troubleshooting

### API won't start
- Check if port 8000 is available
- Ensure virtual environment is activated
- Verify all dependencies are installed

### Models not loading
- First run may take time to download models
- Ensure internet connection for model download
- Check disk space (models ~500MB)

### Frontend can't connect
- Ensure API is running on port 8000
- Check CORS settings in `api/main.py`
- Open browser console for error messages

## 📚 More Information

See README.md for detailed documentation.
