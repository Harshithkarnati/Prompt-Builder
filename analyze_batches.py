import json

# Load all templates
with open('data/templates/templates.json', encoding='utf-8') as f:
    all_templates = json.load(f)

# Load current generated dataset
with open('data/generated_vague/generated_prompts.json', encoding='utf-8') as f:
    generated = json.load(f)

# Get covered template IDs
covered_ids = set(item['template_id'] for item in generated)

# Analyze by category
category_analysis = {}
for template in all_templates:
    cat = template['category']
    if cat not in category_analysis:
        category_analysis[cat] = {
            'total': 0,
            'covered': 0,
            'templates': []
        }
    
    category_analysis[cat]['total'] += 1
    category_analysis[cat]['templates'].append(template)
    
    if template['id'] in covered_ids:
        category_analysis[cat]['covered'] += 1

print("=== DATASET EXPANSION ANALYSIS ===")
print(f"Current dataset: {len(generated)} templates from {len(covered_ids)} IDs")
print(f"Total available: {len(all_templates)} templates")
print(f"Expansion potential: {len(all_templates) - len(covered_ids)} templates")

print("\n=== CATEGORY BREAKDOWN ===")
uncovered_categories = []
for cat, data in category_analysis.items():
    coverage = data['covered'] / data['total'] * 100 if data['total'] > 0 else 0
    print(f"{cat}: {data['covered']}/{data['total']} covered ({coverage:.1f}%)")
    
    if data['covered'] == 0 and data['total'] > 5:  # Focus on substantial uncovered categories
        uncovered_categories.append((cat, data['total']))

print(f"\n=== TOP BATCH OPPORTUNITIES ===")
print("Categories with 0% coverage and 5+ templates:")
for cat, count in sorted(uncovered_categories, key=lambda x: x[1], reverse=True):
    print(f"- {cat}: {count} templates (0% covered)")
    
print(f"\n=== RECOMMENDED BATCH SIZES ===")
print("For comprehensive coverage, create 8-10 batches of 40-60 templates each:")
print("- Batch 2: Business & Leadership (~50 templates)")  
print("- Batch 3: Technology & Engineering (~50 templates)")
print("- Batch 4: Marketing & Sales (~35 templates)")
print("- Batch 5: Health & Science (~45 templates)")
print("- Batch 6: Creative & Personal (~35 templates)")
print("- Batch 7: Professional Development (~40 templates)")
print("- Batch 8: Specialized Domains (~40 templates)")
print("- Batch 9: Remaining categories (~35 templates)")

print(f"\nThis would create a dataset of ~400-450 templates covering all major domains!")