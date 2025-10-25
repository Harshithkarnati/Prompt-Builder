#!/usr/bin/env python3
"""
T5 Dataset Quality Validator

This script validates and analyzes the quality of the generated T5 fine-tuning dataset.
It provides statistics and quality metrics to help assess the dataset.

Author: AI Assistant
Date: October 2025
"""

import json
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any


class T5DatasetValidator:
    """Validate and analyze T5 fine-tuning dataset quality."""
    
    def __init__(self, dataset_file: str):
        """Initialize the validator with dataset file path."""
        self.dataset_file = Path(dataset_file)
        self.entries = []
        
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from JSONL file."""
        print(f"Loading dataset from {self.dataset_file}...")
        
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    self.entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num}: {str(e)}")
        
        print(f"Loaded {len(self.entries)} entries successfully.")
    
    def analyze_basic_stats(self) -> Dict[str, Any]:
        """Analyze basic statistics of the dataset."""
        if not self.entries:
            return {"error": "No entries loaded"}
        
        input_lengths = []
        target_lengths = []
        
        for entry in self.entries:
            input_text = entry.get('input_text', '')
            target_text = entry.get('target_text', '')
            
            input_lengths.append(len(input_text))
            target_lengths.append(len(target_text))
        
        stats = {
            "total_entries": len(self.entries),
            "input_stats": {
                "min_length": min(input_lengths) if input_lengths else 0,
                "max_length": max(input_lengths) if input_lengths else 0,
                "avg_length": statistics.mean(input_lengths) if input_lengths else 0,
                "median_length": statistics.median(input_lengths) if input_lengths else 0
            },
            "target_stats": {
                "min_length": min(target_lengths) if target_lengths else 0,
                "max_length": max(target_lengths) if target_lengths else 0,
                "avg_length": statistics.mean(target_lengths) if target_lengths else 0,
                "median_length": statistics.median(target_lengths) if target_lengths else 0
            }
        }
        
        return stats
    
    def analyze_content_types(self) -> Dict[str, Any]:
        """Analyze the distribution of content types in the dataset."""
        content_type_patterns = {
            'social_media': ['social media', 'instagram', 'twitter', 'facebook', 'linkedin'],
            'business': ['business', 'sales', 'marketing', 'strategy', 'professional'],
            'presentation': ['presentation', 'speaking', 'slides', 'deliver'],
            'email': ['email', 'newsletter', 'message'],
            'creative': ['creative', 'story', 'content', 'engaging'],
            'technical': ['technical', 'documentation', 'system', 'api'],
            'general': []  # catch-all
        }
        
        content_distribution = Counter()
        
        for entry in self.entries:
            target_text = entry.get('target_text', '').lower()
            
            # Check for specific content type patterns
            categorized = False
            for content_type, patterns in content_type_patterns.items():
                if content_type == 'general':
                    continue
                    
                if any(pattern in target_text for pattern in patterns):
                    content_distribution[content_type] += 1
                    categorized = True
                    break
            
            if not categorized:
                content_distribution['general'] += 1
        
        return dict(content_distribution)
    
    def analyze_user_types(self) -> Dict[str, Any]:
        """Analyze the distribution of user types in the dataset."""
        user_types = Counter()
        style_preferences = Counter()
        depth_preferences = Counter()
        speaking_abilities = Counter()
        
        for entry in self.entries:
            input_text = entry.get('input_text', '')
            
            # Extract metadata from input text
            if '| User:' in input_text:
                parts = input_text.split('|')
                for part in parts:
                    part = part.strip()
                    if part.startswith('User:'):
                        user_type = part.replace('User:', '').strip()
                        user_types[user_type] += 1
                    elif part.startswith('Style:'):
                        style = part.replace('Style:', '').strip()
                        style_preferences[style] += 1
                    elif part.startswith('Depth:'):
                        depth = part.replace('Depth:', '').strip()
                        depth_preferences[depth] += 1
                    elif part.startswith('Level:'):
                        level = part.replace('Level:', '').strip()
                        speaking_abilities[level] += 1
        
        return {
            "user_types": dict(user_types.most_common()),
            "style_preferences": dict(style_preferences.most_common()),
            "depth_preferences": dict(depth_preferences.most_common()),
            "speaking_abilities": dict(speaking_abilities.most_common())
        }
    
    def analyze_template_quality(self) -> Dict[str, Any]:
        """Analyze the quality of generated templates."""
        quality_metrics = {
            "has_structure": 0,
            "has_guidelines": 0,
            "has_examples": 0,
            "has_clear_instructions": 0,
            "professional_formatting": 0
        }
        
        quality_indicators = {
            "has_structure": ["**", "###", "##", "Framework:", "Structure:", "Guidelines:"],
            "has_guidelines": ["Guidelines:", "Requirements:", "Standards:", "Principles:"],
            "has_examples": ["example", "e.g.", "for instance", "such as"],
            "has_clear_instructions": ["Please", "Create", "Develop", "Generate", "Write"],
            "professional_formatting": ["**", "##", "-", "1.", "2.", "•"]
        }
        
        for entry in self.entries:
            target_text = entry.get('target_text', '')
            
            for metric, indicators in quality_indicators.items():
                if any(indicator in target_text for indicator in indicators):
                    quality_metrics[metric] += 1
        
        # Convert to percentages
        total = len(self.entries)
        quality_percentages = {
            metric: (count / total * 100) if total > 0 else 0
            for metric, count in quality_metrics.items()
        }
        
        return quality_percentages
    
    def find_potential_issues(self) -> List[Dict[str, Any]]:
        """Find potential issues in the dataset."""
        issues = []
        
        for i, entry in enumerate(self.entries):
            input_text = entry.get('input_text', '')
            target_text = entry.get('target_text', '')
            
            # Check for missing fields
            if not input_text or not target_text:
                issues.append({
                    "entry_index": i,
                    "issue": "missing_fields",
                    "description": "Entry has empty input_text or target_text"
                })
            
            # Check for very short targets (likely incomplete)
            if len(target_text) < 100:
                issues.append({
                    "entry_index": i,
                    "issue": "short_target",
                    "description": f"Target text is very short ({len(target_text)} chars)"
                })
            
            # Check for template artifacts
            template_artifacts = ["{", "}", "[placeholder]", "[example]"]
            if any(artifact in target_text for artifact in template_artifacts):
                issues.append({
                    "entry_index": i,
                    "issue": "template_artifacts",
                    "description": "Target text contains unfilled template placeholders"
                })
            
            # Check for very long targets (might be unwieldy)
            if len(target_text) > 3000:
                issues.append({
                    "entry_index": i,
                    "issue": "very_long_target",
                    "description": f"Target text is very long ({len(target_text)} chars)"
                })
        
        return issues
    
    def generate_sample_report(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Generate a sample of entries for manual review."""
        import random
        
        if len(self.entries) < num_samples:
            num_samples = len(self.entries)
        
        sample_entries = random.sample(self.entries, num_samples)
        
        samples = []
        for entry in sample_entries:
            samples.append({
                "input_text": entry.get('input_text', ''),
                "target_text": entry.get('target_text', '')[:500] + "..." if len(entry.get('target_text', '')) > 500 else entry.get('target_text', ''),
                "input_length": len(entry.get('input_text', '')),
                "target_length": len(entry.get('target_text', ''))
            })
        
        return samples
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        print("Generating comprehensive quality report...")
        
        report = {
            "dataset_file": str(self.dataset_file),
            "basic_stats": self.analyze_basic_stats(),
            "content_types": self.analyze_content_types(),
            "user_metadata": self.analyze_user_types(),
            "quality_metrics": self.analyze_template_quality(),
            "potential_issues": self.find_potential_issues(),
            "sample_entries": self.generate_sample_report()
        }
        
        return report
    
    def print_summary_report(self):
        """Print a human-readable summary report."""
        report = self.generate_full_report()
        
        print("\n" + "="*60)
        print("T5 DATASET QUALITY REPORT")
        print("="*60)
        
        # Basic Stats
        basic = report['basic_stats']
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total Entries: {basic['total_entries']:,}")
        print(f"   Input Text - Avg: {basic['input_stats']['avg_length']:.0f} chars")
        print(f"   Target Text - Avg: {basic['target_stats']['avg_length']:.0f} chars")
        print(f"   Input Range: {basic['input_stats']['min_length']} - {basic['input_stats']['max_length']} chars")
        print(f"   Target Range: {basic['target_stats']['min_length']} - {basic['target_stats']['max_length']} chars")
        
        # Content Types
        content = report['content_types']
        print(f"\n🎯 CONTENT TYPE DISTRIBUTION:")
        for content_type, count in sorted(content.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / basic['total_entries'] * 100) if basic['total_entries'] > 0 else 0
            print(f"   {content_type.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
        
        # User Types
        user_data = report['user_metadata']
        print(f"\n👤 USER TYPE DISTRIBUTION:")
        for user_type, count in list(user_data['user_types'].items())[:5]:
            percentage = (count / basic['total_entries'] * 100) if basic['total_entries'] > 0 else 0
            print(f"   {user_type}: {count:,} ({percentage:.1f}%)")
        
        # Quality Metrics
        quality = report['quality_metrics']
        print(f"\n✅ QUALITY METRICS:")
        for metric, percentage in quality.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"   {metric_name}: {percentage:.1f}%")
        
        # Issues
        issues = report['potential_issues']
        issue_counts = Counter(issue['issue'] for issue in issues)
        print(f"\n⚠️  POTENTIAL ISSUES:")
        if issue_counts:
            for issue_type, count in issue_counts.most_common():
                issue_name = issue_type.replace('_', ' ').title()
                print(f"   {issue_name}: {count} entries")
        else:
            print("   No significant issues detected! ✨")
        
        print(f"\n📝 SAMPLE ENTRIES:")
        for i, sample in enumerate(report['sample_entries'][:3], 1):
            print(f"\n   Sample {i}:")
            print(f"   Input: {sample['input_text']}")
            print(f"   Target: {sample['target_text']}")
            print(f"   Lengths: {sample['input_length']} → {sample['target_length']} chars")
        
        print("\n" + "="*60)


def main():
    """Main function to run the dataset validator."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Default to enhanced dataset
    default_file = project_root / "data" / "training" / "enhanced_t5_dataset.jsonl"
    
    import sys
    if len(sys.argv) > 1:
        dataset_file = Path(sys.argv[1])
    else:
        dataset_file = default_file
    
    try:
        validator = T5DatasetValidator(str(dataset_file))
        validator.print_summary_report()
        
        # Option to save detailed report
        save_report = input("\nSave detailed report to JSON? (y/n): ").lower() == 'y'
        if save_report:
            report = validator.generate_full_report()
            report_file = dataset_file.parent / f"{dataset_file.stem}_quality_report.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"Detailed report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()