"""
Quick fix: Change <thinking> to <reasoning> for competition format
"""
import json

print("Fixing tags: <thinking> → <reasoning>")

for filename in ['train.json', 'val.json', 'all_data.json']:
    filepath = f"data/processed/{filename}"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for item in data:
        item['formatted_output'] = item['formatted_output'].replace(
            '<thinking>', '<reasoning>'
        ).replace(
            '</thinking>', '</reasoning>'
        )
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Fixed {filename}")

print("\n✅ All files updated to use <reasoning> tags!")