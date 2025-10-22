from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from typing import List, Optional

class T5PromptOptimizer:
    """Optimizes prompts using T5 model."""
    
    def __init__(self, model_name: str = 'google/flan-t5-small'):
        """
        Initialize T5 optimizer.
        
        Args:
            model_name: Hugging Face model identifier
        """
        print(f"Loading T5 model: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"T5 model loaded on {self.device}")
    
    def optimize_prompt(self, user_prompt: str, 
                       retrieved_prompts: Optional[List[str]] = None,
                       max_length: int = 150,
                       temperature: float = 0.7) -> str:
        """
        Optimize a user prompt using T5 model.
        
        Args:
            user_prompt: Original user prompt
            retrieved_prompts: Similar prompts for context
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Optimized prompt string
        """
        # Build context from retrieved prompts
        context = ""
        if retrieved_prompts:
            context = "Similar prompts:\n" + "\n".join(
                f"- {p}" for p in retrieved_prompts[:3]
            ) + "\n\n"
        
        # Create instruction for T5
        instruction = (
            f"{context}"
            f"Improve and optimize the following prompt to be more clear, "
            f"specific, and effective:\n\n{user_prompt}"
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(
            instruction, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )
        
        optimized = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return optimized.strip()
    
    def generate_variations(self, prompt: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple variations of a prompt.
        
        Args:
            prompt: Input prompt
            num_variations: Number of variations to generate
            
        Returns:
            List of prompt variations
        """
        instruction = f"Rewrite this prompt in different ways: {prompt}"
        
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=150,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=num_variations
        )
        
        variations = [
            self.tokenizer.decode(output, skip_special_tokens=True).strip()
            for output in outputs
        ]
        return variations

# Global optimizer instance
_optimizer = None

def get_optimizer() -> T5PromptOptimizer:
    """Get or create global T5 optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = T5PromptOptimizer()
    return _optimizer

def generate_t5_prompt(user_prompt: str, 
                      retrieved_prompts: Optional[List[str]] = None) -> str:
    """
    Convenience function to generate optimized prompt.
    
    Args:
        user_prompt: Original user prompt
        retrieved_prompts: Similar prompts for context
        
    Returns:
        Optimized prompt
    """
    optimizer = get_optimizer()
    return optimizer.optimize_prompt(user_prompt, retrieved_prompts)

if __name__ == "__main__":
    # Test T5 optimization
    test_prompt = "write code"
    print(f"\nOriginal: {test_prompt}")
    optimized = generate_t5_prompt(test_prompt)
    print(f"Optimized: {optimized}")

