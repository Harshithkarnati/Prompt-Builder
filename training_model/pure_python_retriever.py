#!/usr/bin/env python3
"""
Pure Python Template Retriever - Option 2 Modern Solution
===========================================================

A self-contained template retrieval system using only Python standard library.
No external dependencies required (no sklearn, no sentence-transformers, no chromadb).

Features:
- TF-IDF vectorization from scratch
- Cosine similarity calculation
- Multiple normalization methods (softmax, min-max, percentage)
- Batch processing with progress tracking
- Comprehensive error handling and validation
- Production-ready for 1000+ prompts

Author: GitHub Copilot Assistant
Date: October 2025
"""

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class TemplateMatch:
    """Represents a template match with relevance score."""
    template_name: str
    template_text: str
    relevance_score: float
    similarity_raw: float


@dataclass
class RetrievalResult:
    """Represents retrieval results for a single vague prompt."""
    vague_prompt_text: str
    top_matches: List[TemplateMatch]
    processing_time_ms: float


class PurePythonTemplateRetriever:
    """
    Pure Python implementation of template retrieval using TF-IDF and cosine similarity.
    
    This class provides a complete semantic matching solution without any external
    dependencies, making it highly portable and environment-friendly.
    """
    
    def __init__(self, max_features: int = 1000):
        """
        Initialize the retriever.
        
        Args:
            max_features: Maximum number of TF-IDF features to use
        """
        self.max_features = max_features
        self.templates = []
        self.template_vectors = []
        self.vocabulary = {}
        self.idf_scores = {}
        self.is_fitted = False
        
        print("✅ Initialized PurePythonTemplateRetriever")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Enhanced text preprocessing for better semantic matching.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of cleaned tokens with better semantic features
        """
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        
        # Replace common punctuation with spaces for better tokenization
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Extract words (more permissive pattern)
        tokens = re.findall(r'\b\w{2,}\b', text)  # At least 2 characters
        
        # Enhanced stopword removal (more comprehensive)
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'would', 'you', 'your', 'i',
            'me', 'my', 'we', 'our', 'they', 'them', 'their', 'this', 'that',
            'can', 'could', 'should', 'may', 'might', 'must', 'shall', 'do',
            'does', 'did', 'have', 'had', 'been', 'being', 'get', 'got', 'go',
            'goes', 'went', 'come', 'came', 'take', 'took', 'make', 'made',
            'see', 'saw', 'know', 'knew', 'think', 'thought', 'say', 'said',
            'tell', 'told', 'ask', 'asked', 'give', 'gave', 'put', 'let',
            'also', 'just', 'only', 'even', 'still', 'now', 'then', 'here',
            'there', 'where', 'when', 'why', 'how', 'what', 'who', 'which',
            'some', 'any', 'all', 'each', 'every', 'other', 'another', 'such',
            'like', 'than', 'so', 'very', 'more', 'most', 'much', 'many'
        }
        
        # Filter out stopwords and very short tokens
        filtered_tokens = [token for token in tokens 
                          if len(token) > 2 and token not in stopwords]
        
        # Add simple bigrams for better context
        bigrams = []
        for i in range(len(filtered_tokens) - 1):
            bigram = f"{filtered_tokens[i]}_{filtered_tokens[i+1]}"
            bigrams.append(bigram)
        
        # Combine unigrams and bigrams
        all_tokens = filtered_tokens + bigrams
        
        return all_tokens
    
    def _build_vocabulary(self, documents: List[str]) -> None:
        """
        Build vocabulary with improved term selection for better semantic matching.
        
        Args:
            documents: List of documents to build vocabulary from
        """
        print("🔄 Building vocabulary and calculating IDF scores...")
        
        # Count document frequency for each term
        doc_freq = defaultdict(int)
        term_total_freq = defaultdict(int)
        total_docs = len(documents)
        
        # Process each document
        for doc in documents:
            tokens = self._preprocess_text(doc)
            unique_tokens = set(tokens)
            
            for token in tokens:
                term_total_freq[token] += 1
            
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Filter terms: must appear in at least 2 documents but not more than 80% of documents
        min_doc_freq = max(2, total_docs // 100)  # At least 2 docs or 1% of docs
        max_doc_freq = int(total_docs * 0.8)  # Not more than 80% of docs
        
        filtered_terms = {
            term: freq for term, freq in doc_freq.items()
            if min_doc_freq <= freq <= max_doc_freq and len(term) >= 3
        }
        
        # Score terms by combining document frequency and total frequency
        # Prefer terms that are discriminative (medium doc freq) but appear often
        term_scores = {}
        for term, doc_freq_val in filtered_terms.items():
            total_freq_val = term_total_freq[term]
            # Balance between being discriminative and being frequent
            discriminative_score = doc_freq_val / total_docs
            frequency_score = total_freq_val / sum(term_total_freq.values())
            combined_score = discriminative_score * frequency_score
            term_scores[term] = combined_score
        
        # Select top features by combined score
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        selected_terms = sorted_terms[:self.max_features]
        
        # Create vocabulary mapping
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(selected_terms)}
        
        # Calculate IDF scores with smoothing
        self.idf_scores = {}
        for term, _ in selected_terms:
            doc_freq_val = doc_freq[term]
            # IDF with smoothing: log((N + 1) / (df + 1)) + 1
            idf = math.log((total_docs + 1) / (doc_freq_val + 1)) + 1
            self.idf_scores[term] = idf
        
        print(f"✅ Built vocabulary with {len(self.vocabulary)} features")
        print(f"📊 Filtered from {len(doc_freq)} total terms to {len(self.vocabulary)} discriminative terms")
    
    def _compute_tf_vector(self, text: str) -> List[float]:
        """
        Compute enhanced TF (Term Frequency) vector with better weighting.
        
        Args:
            text: Input text
            
        Returns:
            TF vector as list of floats
        """
        tokens = self._preprocess_text(text)
        
        # Count term frequencies
        tf_counts = Counter(tokens)
        total_terms = len(tokens)
        
        # Create TF vector with log normalization
        tf_vector = [0.0] * len(self.vocabulary)
        
        for term, count in tf_counts.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                # Use log normalization: 1 + log(tf) if tf > 0, else 0
                if count > 0:
                    tf_vector[idx] = 1 + math.log(count)
                else:
                    tf_vector[idx] = 0.0
        
        return tf_vector
    
    def _compute_tfidf_vector(self, text: str) -> List[float]:
        """
        Compute TF-IDF vector for a document.
        
        Args:
            text: Input text
            
        Returns:
            TF-IDF vector as list of floats
        """
        tf_vector = self._compute_tf_vector(text)
        
        # Apply IDF weights
        tfidf_vector = []
        for i, tf_score in enumerate(tf_vector):
            term = list(self.vocabulary.keys())[i]
            idf_score = self.idf_scores.get(term, 0.0)
            tfidf_score = tf_score * idf_score
            tfidf_vector.append(tfidf_score)
        
        return tfidf_vector
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Compute magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        # Handle zero magnitude case
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        # Compute cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, similarity))
    
    def _normalize_scores(self, scores: List[float], method: str = "softmax") -> List[float]:
        """
        Normalize relevance scores to [0, 1] range.
        
        Args:
            scores: Raw similarity scores
            method: Normalization method ("softmax", "min_max", or "percentage")
            
        Returns:
            Normalized scores
        """
        if not scores or all(s == 0 for s in scores):
            return [0.0] * len(scores)
        
        if method == "softmax":
            # Softmax normalization (emphasizes differences)
            max_score = max(scores)
            # Subtract max for numerical stability
            exp_scores = [math.exp(s - max_score) for s in scores]
            sum_exp = sum(exp_scores)
            
            if sum_exp == 0:
                return [1.0 / len(scores)] * len(scores)
            
            return [exp_s / sum_exp for exp_s in exp_scores]
        
        elif method == "min_max":
            # Min-max normalization (preserves relative differences)
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                return [1.0 / len(scores)] * len(scores)
            
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        elif method == "percentage":
            # Percentage normalization (original approach)
            total_score = sum(scores)
            
            if total_score == 0:
                return [1.0 / len(scores)] * len(scores)
            
            return [s / total_score for s in scores]
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def load_templates(self, templates_file: Union[str, Path]) -> int:
        """
        Load templates from JSON file and compute TF-IDF vectors.
        
        Args:
            templates_file: Path to templates JSON file
            
        Returns:
            Number of templates loaded
        """
        print(f"📚 Loading templates from {templates_file}")
        
        try:
            with open(templates_file, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            
            # Extract templates
            self.templates = []
            template_texts = []
            
            for template in templates_data:
                if isinstance(template, dict):
                    # Handle the actual template structure
                    name = template.get('template_name', template.get('name', 'Unknown'))
                    
                    # Extract and combine relevant content for better matching
                    description = template.get('template_description', '')
                    format_content = template.get('template_format', template.get('content', ''))
                    category = template.get('category', '')
                    use_case = template.get('example_use_case', '')
                    
                    # Create comprehensive content for matching
                    # Prioritize description and use case for semantic understanding
                    content_parts = []
                    if description:
                        content_parts.append(description)
                    if use_case:
                        content_parts.append(use_case)
                    if category:
                        content_parts.append(category)
                    if format_content:
                        # Extract key phrases from template format (remove placeholder syntax)
                        clean_format = re.sub(r'\{[^}]+\}', '', format_content)
                        clean_format = re.sub(r'##+\s*', '', clean_format)  # Remove markdown headers
                        content_parts.append(clean_format)
                    
                    content = ' '.join(content_parts)
                    
                    self.templates.append({
                        'name': name,
                        'content': content,
                        'original_format': format_content
                    })
                    template_texts.append(content)
                else:
                    # Handle simple string templates
                    template_texts.append(str(template))
                    self.templates.append({
                        'name': f'Template_{len(self.templates)}',
                        'content': str(template),
                        'original_format': str(template)
                    })
            
            print(f"✅ Loaded {len(self.templates)} templates")
            
            # Build vocabulary and compute vectors
            print("🔄 Computing TF-IDF vectors...")
            self._build_vocabulary(template_texts)
            
            # Compute TF-IDF vectors for all templates
            self.template_vectors = []
            for text in template_texts:
                vector = self._compute_tfidf_vector(text)
                self.template_vectors.append(vector)
            
            print(f"✅ Computed {len(self.template_vectors)} template vectors")
            self.is_fitted = True
            
            return len(self.templates)
            
        except Exception as e:
            print(f"❌ Error loading templates: {str(e)}")
            raise
    
    def retrieve_templates(self, vague_prompt: str, top_k: int = 3, 
                          normalize_method: str = "softmax") -> List[TemplateMatch]:
        """
        Retrieve top-k most relevant templates for a vague prompt.
        
        Args:
            vague_prompt: The vague prompt to match
            top_k: Number of top templates to return
            normalize_method: Score normalization method
            
        Returns:
            List of TemplateMatch objects
        """
        if not self.is_fitted:
            raise ValueError("Retriever not fitted. Call load_templates() first.")
        
        if not vague_prompt or not isinstance(vague_prompt, str):
            return []
        
        # Compute TF-IDF vector for vague prompt
        prompt_vector = self._compute_tfidf_vector(vague_prompt)
        
        # Compute similarities with all templates
        similarities = []
        for template_vector in self.template_vectors:
            similarity = self._cosine_similarity(prompt_vector, template_vector)
            similarities.append(similarity)
        
        if not similarities:
            return []
        
        # Get top-k template indices
        indexed_similarities = list(enumerate(similarities))
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_similarities[:top_k]
        
        # Extract top scores and normalize
        top_scores = [similarities[idx] for idx, _ in top_indices]
        normalized_scores = self._normalize_scores(top_scores, normalize_method)
        
        # Create TemplateMatch objects
        matches = []
        for i, (template_idx, raw_score) in enumerate(top_indices):
            template = self.templates[template_idx]
            match = TemplateMatch(
                template_name=template['name'],
                template_text=template.get('original_format', template['content']),  # Use original format
                relevance_score=normalized_scores[i],
                similarity_raw=raw_score
            )
            matches.append(match)
        
        return matches
    
    def process_vague_prompts(self, vague_prompts: List[str], output_file: Union[str, Path],
                            top_k: int = 3, normalize_method: str = "softmax") -> List[RetrievalResult]:
        """
        Process multiple vague prompts and save results.
        
        Args:
            vague_prompts: List of vague prompts to process
            output_file: Path to save results
            top_k: Number of top templates per prompt
            normalize_method: Score normalization method
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.is_fitted:
            raise ValueError("Retriever not fitted. Call load_templates() first.")
        
        print(f"🚀 Processing {len(vague_prompts)} vague prompts...")
        
        results = []
        start_time = time.time()
        
        # Process each prompt with progress tracking
        for i, prompt in enumerate(vague_prompts):
            # Progress update every 50 prompts
            if i % 50 == 0:
                print(f"📝 Processed {i}/{len(vague_prompts)} prompts...")
            
            prompt_start = time.time()
            
            try:
                # Retrieve templates
                matches = self.retrieve_templates(prompt, top_k, normalize_method)
                
                # Calculate processing time
                processing_time = (time.time() - prompt_start) * 1000  # ms
                
                # Create result
                result = RetrievalResult(
                    vague_prompt_text=prompt,
                    top_matches=matches,
                    processing_time_ms=processing_time
                )
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error processing prompt {i}: {str(e)}")
                # Create empty result for failed prompts
                result = RetrievalResult(
                    vague_prompt_text=prompt,
                    top_matches=[],
                    processing_time_ms=0.0
                )
                results.append(result)
        
        # Save results to JSONL file with proper formatting
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    # Convert to T5 training format with proper structure
                    output_entry = {
                        "vague_prompt": result.vague_prompt_text,
                        "template_matches": [
                            {
                                "template_name": match.template_name,
                                "template_content": match.template_text,
                                "relevance_score": match.relevance_score
                            }
                            for match in result.top_matches
                        ],
                        "processing_time_ms": result.processing_time_ms
                    }
                    # Write with proper indentation for readability
                    f.write(json.dumps(output_entry, ensure_ascii=False, indent=2) + '\n')
            
        except Exception as e:
            print(f"⚠️ Error saving results: {str(e)}")
        
        total_time = time.time() - start_time
        avg_time = (total_time / len(vague_prompts)) * 1000 if vague_prompts else 0
        
        print(f"✅ Successfully processed {len(results)} prompts")
        print(f"⏱️  Average processing time: {avg_time:.2f}ms per prompt")
        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        
        return results


def main():
    """Main function to demonstrate usage."""
    print("🚀 Pure Python Template Retriever")
    print("=" * 50)
    print("✅ No external dependencies required!")
    
    # Configuration
    config = {
        "templates_file": "../data/templates/templates.json",
        "vague_prompts_file": "../data/generated_vague/generated_vague_prompts.json",
        "output_file": "../data/retrieval/pure_python_results.jsonl",
        "normalize_method": "softmax",
        "top_k": 3,
        "max_features": 1000
    }
    
    # Initialize retriever
    retriever = PurePythonTemplateRetriever(
        max_features=config["max_features"]
    )
    
    # Load templates
    templates_count = retriever.load_templates(config["templates_file"])
    print(f"📚 Loaded {templates_count} templates")
    
    # Load vague prompts (limit for testing)
    print(f"📝 Loading vague prompts from {config['vague_prompts_file']}")
    with open(config["vague_prompts_file"], 'r', encoding='utf-8') as f:
        vague_prompts_data = json.load(f)
    
    # Extract the actual prompt text
    vague_prompts = []
    for prompt_obj in vague_prompts_data:
        if isinstance(prompt_obj, dict):
            prompt_text = prompt_obj.get('user_vague_prompt', prompt_obj.get('prompt', ''))
            if prompt_text:
                vague_prompts.append(prompt_text)
        else:
            vague_prompts.append(str(prompt_obj))
    
    # Process all prompts (remove test limit)
    print(f"📝 Processing all {len(vague_prompts)} vague prompts")
    
    # Process prompts
    results = retriever.process_vague_prompts(vague_prompts, config["output_file"])
    
    # Print summary statistics
    print("\n📊 Processing Summary")
    print("=" * 50)
    
    successful_results = [r for r in results if r.top_matches]
    print(f"✅ Successfully processed: {len(successful_results)} prompts")
    
    if successful_results:
        processing_times = [r.processing_time_ms for r in successful_results]
        avg_time = sum(processing_times) / len(processing_times)
        total_time = sum(processing_times) / 1000
        print(f"⏱️  Average processing time: {avg_time:.2f}ms per prompt")
        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        
        # Collect all relevance scores for validation
        all_scores = []
        for result in successful_results:
            for match in result.top_matches:
                all_scores.append(match.relevance_score)
        
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            avg_score = sum(all_scores) / len(all_scores)
            print(f"🎯 Relevance scores - Min: {min_score:.3f}, Max: {max_score:.3f}, Avg: {avg_score:.3f}")
            
            # Validate all scores are in valid range
            valid_scores = all(0 <= s <= 1 for s in all_scores)
            print(f"✅ All scores valid (0-1): {valid_scores}")
        
        # Show sample results
        print(f"\n📋 Sample Results:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. '{result.vague_prompt_text[:50]}...'")
            for match in result.top_matches:
                print(f"     → {match.template_name}: {match.relevance_score:.3f}")
    
    print(f"\n💾 Results saved to: {config['output_file']}")
    print("🎉 Ready for T5 model training!")


if __name__ == "__main__":
    main()