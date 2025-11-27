"""
Step 2: Convert datasets to competition format
Input:  Raw datasets with various answer formats
Output: Unified format with <thinking> and <answer> tags

Run: python src/step2_format_data.py
"""

from datasets import load_from_disk, Dataset
import re
import json
import os

os.makedirs("data/processed", exist_ok=True)

print("="*60)
print("STEP 2: FORMATTING DATA FOR COMPETITION")
print("="*60)

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_gsm8k_answer(answer_text):
    """
    GSM8K format: reasoning text ending with #### [number]
    Example: "Natalia sold 48/2 = 24 clips... #### 72"
    """
    # Find the #### marker and extract the number after it
    match = re.search(r'####\s*(\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
    if match:
        # Remove commas from numbers like "1,000"
        return match.group(1).replace(',', '')
    return None

def extract_gsm8k_reasoning(answer_text):
    """Extract the reasoning part (everything before ####)"""
    parts = answer_text.split('####')
    if len(parts) >= 1:
        return parts[0].strip()
    return answer_text.strip()

def format_for_competition(question, reasoning, answer):
    """
    Convert to competition format:
    <thinking>reasoning</thinking>
    <answer>answer</answer>
    """
    formatted = f"""<thinking>
{reasoning}
</thinking>
<answer>{answer}</answer>"""
    return formatted

# ============================================
# PROCESS GSM8K
# ============================================
print("\n📦 [1/2] Processing GSM8K...")

try:
    gsm8k = load_from_disk("data/raw/gsm8k")
    
    formatted_examples = []
    skipped = 0
    
    for example in gsm8k['train']:
        question = example['question']
        raw_answer = example['answer']
        
        # Extract reasoning and final answer
        reasoning = extract_gsm8k_reasoning(raw_answer)
        final_answer = extract_gsm8k_answer(raw_answer)
        
        if final_answer is None:
            skipped += 1
            continue
        
        # Create formatted example
        formatted_examples.append({
            'question': question,
            'reasoning': reasoning,
            'answer': final_answer,
            'formatted_output': format_for_competition(question, reasoning, final_answer),
            'source': 'gsm8k'
        })
    
    print(f"✅ GSM8K processed: {len(formatted_examples):,} examples")
    print(f"   Skipped (no answer found): {skipped}")
    
    # Show example
    print("\n" + "-"*60)
    print("📝 FORMATTED EXAMPLE (GSM8K):")
    print("-"*60)
    ex = formatted_examples[0]
    print(f"QUESTION:\n{ex['question']}\n")
    print(f"FORMATTED OUTPUT:\n{ex['formatted_output']}")
    print("-"*60)
    
    gsm8k_formatted = formatted_examples
    
except Exception as e:
    print(f"❌ Error processing GSM8K: {e}")
    gsm8k_formatted = []

# ============================================
# PROCESS ORCA-MATH
# ============================================
print("\n📦 [2/2] Processing Orca-Math...")

try:
    orca = load_from_disk("data/raw/orca_math")
    
    formatted_examples = []
    
    # Only take first 50,000 to keep training manageable
    # (200K is too much for a 9-hour TPU session)
    MAX_ORCA = 50000
    
    for i, example in enumerate(orca['train']):
        if i >= MAX_ORCA:
            break
            
        question = example['question']
        raw_answer = example['answer']
        
        # Orca-Math answers are already reasoning + answer
        # We'll use the full answer as reasoning and try to extract final number
        reasoning = raw_answer.strip()
        
        # Try to find the last number in the answer (usually the final answer)
        numbers = re.findall(r'[\d,]+(?:\.\d+)?', raw_answer)
        if numbers:
            # Take the last number as the answer
            final_answer = numbers[-1].replace(',', '')
        else:
            final_answer = "See reasoning above"
        
        formatted_examples.append({
            'question': question,
            'reasoning': reasoning,
            'answer': final_answer,
            'formatted_output': format_for_competition(question, reasoning, final_answer),
            'source': 'orca_math'
        })
    
    print(f"✅ Orca-Math processed: {len(formatted_examples):,} examples (limited from 200K)")
    
    # Show example
    print("\n" + "-"*60)
    print("📝 FORMATTED EXAMPLE (Orca-Math):")
    print("-"*60)
    ex = formatted_examples[0]
    print(f"QUESTION:\n{ex['question']}\n")
    print(f"FORMATTED OUTPUT:\n{ex['formatted_output'][:500]}...")
    print("-"*60)
    
    orca_formatted = formatted_examples
    
except Exception as e:
    print(f"❌ Error processing Orca-Math: {e}")
    orca_formatted = []

# ============================================
# COMBINE AND SAVE
# ============================================
print("\n💾 Combining and saving...")

all_examples = gsm8k_formatted + orca_formatted

# Shuffle the data
import random
random.seed(42)
random.shuffle(all_examples)

# Split into train/validation (95/5)
split_idx = int(len(all_examples) * 0.95)
train_data = all_examples[:split_idx]
val_data = all_examples[split_idx:]

# Save as JSON (easy to load later)
with open("data/processed/train.json", 'w') as f:
    json.dump(train_data, f, indent=2)
    
with open("data/processed/val.json", 'w') as f:
    json.dump(val_data, f, indent=2)

# Also save as single combined file
with open("data/processed/all_data.json", 'w') as f:
    json.dump(all_examples, f, indent=2)

print(f"✅ Saved to data/processed/")
print(f"   train.json: {len(train_data):,} examples")
print(f"   val.json:   {len(val_data):,} examples")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("✅ STEP 2 COMPLETE: DATA FORMATTED")
print("="*60)
print(f"""
Dataset Statistics:
├── GSM8K:     {len(gsm8k_formatted):,} examples
├── Orca-Math: {len(orca_formatted):,} examples
└── Total:     {len(all_examples):,} examples

Output Format:
<thinking>
[step-by-step reasoning]
</thinking>
<answer>[final answer]</answer>

Files saved:
├── data/processed/train.json ({len(train_data):,} examples)
├── data/processed/val.json   ({len(val_data):,} examples)
└── data/processed/all_data.json

Next: Step 3 - Create Kaggle notebook for training
""")