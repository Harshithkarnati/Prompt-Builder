import json
from pathlib import Path
import re

base = Path(__file__).resolve().parents[1]
vg_path = base / 'data' / 'generated_vague' / 'generated_vague_prompts.json'
pp_path = base / 'data' / 'retrieval' / 'pure_python_results.jsonl'

# Load vague prompts mapping
with vg_path.open('r', encoding='utf-8') as f:
    vague_list = json.load(f)

# Build mapping from prompt string to metadata
mapping = {}
for item in vague_list:
    prompt = item.get('user_vague_prompt') or item.get('vague_prompt')
    if prompt:
        mapping[prompt.strip()] = {
            'user_type': item.get('user_type'),
            'style_preference': item.get('style_preference'),
            'depth_preference': item.get('depth_preference'),
            'speaking_ability': item.get('speaking_ability')
        }

print(f"Created mapping for {len(mapping)} vague prompts")

# Read the entire file content and split by JSON objects
with pp_path.open('r', encoding='utf-8') as f:
    content = f.read()

# Split content by }{ pattern to separate JSON objects
json_objects = []
if content.strip():
    # Add missing braces and split
    if not content.startswith('['):
        # Split by '}' followed by optional whitespace and '{'
        parts = re.split(r'}\s*{', content)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # Add missing braces
            if i == 0:
                part = part + '}'
            elif i == len(parts) - 1:
                part = '{' + part
            else:
                part = '{' + part + '}'
            
            try:
                obj = json.loads(part)
                json_objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"Failed to parse object {i}: {e}")
                print(f"Content preview: {part[:100]}...")

print(f"Found {len(json_objects)} JSON objects")

# Update objects
updated_count = 0
for obj in json_objects:
    vague = obj.get('vague_prompt') or obj.get('user_vague_prompt')
    if vague:
        meta = mapping.get(vague.strip())
        if meta:
            # add new fields if missing
            obj['user_type'] = meta.get('user_type')
            obj['style_preference'] = meta.get('style_preference')
            obj['depth_preference'] = meta.get('depth_preference')
            obj['speaking_ability'] = meta.get('speaking_ability')
            updated_count += 1
    
    # remove processing_time fields if present
    for key in ['processing_time_ms', 'processing_time']:
        if key in obj:
            del obj[key]

print(f"Updated {updated_count} objects with vague prompt metadata")

# Write back as proper JSONL (one JSON object per line)
with pp_path.open('w', encoding='utf-8') as f:
    for obj in json_objects:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"Updated {pp_path} — wrote {len(json_objects)} lines")
