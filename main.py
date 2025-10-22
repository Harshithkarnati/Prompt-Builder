#!/usr/bin/env python3
"""
Prompt Builder - AI-powered prompt optimization system
Main entry point for the application
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Prompt Builder - AI-powered prompt optimization"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Start the API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    api_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    api_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw prompts')
    preprocess_parser.add_argument('--input', default='data/raw_prompts.json', 
                                   help='Input raw prompts file')
    preprocess_parser.add_argument('--output', default='data/processed_prompts.json',
                                   help='Output processed prompts file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the system with a sample prompt')
    test_parser.add_argument('prompt', nargs='?', default='Write a function to sort an array',
                           help='Test prompt')
    
    args = parser.parse_args()
    
    if args.command == 'api':
        # Start API server
        import uvicorn
        from api.main import app
        
        print(f"Starting Prompt Builder API on {args.host}:{args.port}")
        print(f"API docs will be available at http://{args.host}:{args.port}/docs")
        
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    
    elif args.command == 'preprocess':
        # Run preprocessing
        from utils.preprocessing import load_and_preprocess
        
        print(f"Preprocessing prompts from {args.input}")
        load_and_preprocess(args.input, args.output)
        print("Done!")
    
    elif args.command == 'test':
        # Test the system
        print("Testing Prompt Builder...")
        print(f"Input prompt: {args.prompt}\n")
        
        from utils.retrieval import retrieve_prompts
        from models.t5_prompt_optimizer.t5 import generate_t5_prompt
        
        print("Step 1: Retrieving similar prompts...")
        retrieved = retrieve_prompts(args.prompt, top_k=3)
        for i, prompt in enumerate(retrieved, 1):
            print(f"  {i}. {prompt}")
        
        print("\nStep 2: Optimizing prompt with T5...")
        optimized = generate_t5_prompt(args.prompt, retrieved)
        print(f"\nOptimized prompt:\n{optimized}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
