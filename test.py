"""
Test script for Prompt Builder
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from utils.retrieval import retrieve_prompts, get_retriever
        from models.t5_prompt_optimizer.t5 import generate_t5_prompt, get_optimizer
        from utils.preprocessing import preprocess_prompts
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_retrieval():
    """Test prompt retrieval."""
    print("\nTesting retrieval system...")
    try:
        from utils.retrieval import retrieve_prompts
        
        test_query = "Write code to sort data"
        results = retrieve_prompts(test_query, top_k=3)
        
        print(f"Query: {test_query}")
        print(f"Retrieved {len(results)} prompts:")
        for i, prompt in enumerate(results, 1):
            print(f"  {i}. {prompt}")
        
        assert len(results) > 0, "No results returned"
        print("✓ Retrieval test passed")
        return True
    except Exception as e:
        print(f"✗ Retrieval test failed: {e}")
        return False

def test_optimization():
    """Test T5 optimization."""
    print("\nTesting T5 optimization...")
    try:
        from models.t5_prompt_optimizer.t5 import generate_t5_prompt
        
        test_prompt = "write code"
        retrieved = ["Generate a Python function", "Write a program", "Create a script"]
        
        optimized = generate_t5_prompt(test_prompt, retrieved)
        
        print(f"Original: {test_prompt}")
        print(f"Optimized: {optimized}")
        
        assert len(optimized) > 0, "Empty optimization result"
        print("✓ Optimization test passed")
        return True
    except Exception as e:
        print(f"✗ Optimization test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing."""
    print("\nTesting preprocessing...")
    try:
        from utils.preprocessing import clean_text, preprocess_prompts
        
        test_text = "  Write   a   function!!!  "
        cleaned = clean_text(test_text)
        
        print(f"Original: '{test_text}'")
        print(f"Cleaned: '{cleaned}'")
        
        assert len(cleaned) > 0, "Empty result"
        print("✓ Preprocessing test passed")
        return True
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False

def test_full_pipeline():
    """Test complete pipeline."""
    print("\nTesting full pipeline...")
    try:
        from utils.retrieval import retrieve_prompts
        from models.t5_prompt_optimizer.t5 import generate_t5_prompt
        
        user_prompt = "explain AI concepts"
        
        # Step 1: Retrieve
        print("Step 1: Retrieving similar prompts...")
        retrieved = retrieve_prompts(user_prompt, top_k=3)
        print(f"  Retrieved {len(retrieved)} prompts")
        
        # Step 2: Optimize
        print("Step 2: Optimizing with T5...")
        optimized = generate_t5_prompt(user_prompt, retrieved)
        print(f"  Optimized prompt length: {len(optimized)}")
        
        print("\n📝 Results:")
        print(f"Original: {user_prompt}")
        print(f"Optimized: {optimized}")
        
        assert len(optimized) > 0, "Empty optimization"
        print("\n✓ Full pipeline test passed")
        return True
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Prompt Builder Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Preprocessing", test_preprocessing),
        ("Retrieval", test_retrieval),
        ("Optimization", test_optimization),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
