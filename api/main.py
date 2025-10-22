from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.retrieval import retrieve_prompts, get_retriever
from models.t5_prompt_optimizer.t5 import generate_t5_prompt, get_optimizer

# Initialize FastAPI app
app = FastAPI(
    title="Prompt Builder API",
    description="AI-powered prompt optimization and generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PromptRequest(BaseModel):
    user_prompt: str = Field(..., description="User's input prompt to optimize")
    top_k: Optional[int] = Field(3, description="Number of similar prompts to retrieve")
    include_variations: Optional[bool] = Field(False, description="Generate prompt variations")

class PromptResponse(BaseModel):
    original_prompt: str
    optimized_prompt: str
    retrieved_prompts: List[str]
    variations: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models when API starts."""
    try:
        print("Initializing Prompt Builder API...")
        # Warm up retriever
        get_retriever()
        # Warm up T5 optimizer
        get_optimizer()
        print("API ready!")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

@app.get("/", response_model=HealthResponse)
def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Prompt Builder API is running"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Detailed health check."""
    try:
        # Check if models are loaded
        get_retriever()
        get_optimizer()
        return {
            "status": "healthy",
            "message": "All systems operational"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/generate_prompt", response_model=PromptResponse)
async def generate_prompt(req: PromptRequest):
    """
    Generate an optimized prompt from user input.
    
    Steps:
    1. Retrieve similar prompts using semantic search
    2. Optimize prompt using T5 model with retrieved context
    3. Optionally generate variations
    """
    try:
        # Step 1: Retrieve similar prompts
        print(f"Retrieving similar prompts for: {req.user_prompt}")
        retrieved = retrieve_prompts(req.user_prompt, top_k=req.top_k)
        
        # Step 2: Optimize with T5
        print("Optimizing prompt with T5...")
        optimized = generate_t5_prompt(req.user_prompt, retrieved)
        
        # Step 3: Generate variations if requested
        variations = None
        if req.include_variations:
            print("Generating prompt variations...")
            optimizer = get_optimizer()
            variations = optimizer.generate_variations(optimized, num_variations=3)
        
        return {
            "original_prompt": req.user_prompt,
            "optimized_prompt": optimized,
            "retrieved_prompts": retrieved,
            "variations": variations
        }
    
    except Exception as e:
        print(f"Error generating prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")

@app.post("/retrieve")
async def retrieve_similar_prompts(req: PromptRequest):
    """Retrieve similar prompts only."""
    try:
        retrieved = retrieve_prompts(req.user_prompt, top_k=req.top_k)
        return {
            "query": req.user_prompt,
            "similar_prompts": retrieved
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving prompts: {str(e)}")

@app.post("/optimize")
async def optimize_only(req: PromptRequest):
    """Optimize prompt without retrieval."""
    try:
        optimized = generate_t5_prompt(req.user_prompt)
        return {
            "original_prompt": req.user_prompt,
            "optimized_prompt": optimized
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing prompt: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
