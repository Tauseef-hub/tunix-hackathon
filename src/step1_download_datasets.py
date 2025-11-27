"""
Step 1: Download FREE datasets with reasoning traces
No API costs - everything from HuggingFace

Run: python src/step1_download_datasets.py
"""

from datasets import load_dataset
import json
import os

# Create data directories if they don't exist
os.makedirs("data/raw", exist_ok=True)

print("="*60)
print("STEP 1: DOWNLOADING FREE REASONING DATASETS")
print("="*60)

# ============================================
# DATASET 1: GSM8K (Grade School Math)
# 8,792 problems with step-by-step solutions
# ============================================
print("\n📥 [1/3] Downloading GSM8K...")
try:
    gsm8k = load_dataset("openai/gsm8k", "main")
    
    print(f"✅ GSM8K loaded!")
    print(f"   Train: {len(gsm8k['train']):,} examples")
    print(f"   Test:  {len(gsm8k['test']):,} examples")
    
    # Show example
    print("\n" + "-"*60)
    print("📝 EXAMPLE FROM GSM8K:")
    print("-"*60)
    example = gsm8k['train'][0]
    print(f"QUESTION:\n{example['question']}\n")
    print(f"ANSWER (with steps):\n{example['answer']}")
    print("-"*60)
    
    # Save to disk
    gsm8k.save_to_disk("data/raw/gsm8k")
    print(f"💾 Saved to data/raw/gsm8k")
    
except Exception as e:
    print(f"❌ Error loading GSM8K: {e}")

# ============================================
# DATASET 2: MATH (Competition Level)
# 12,500 problems with detailed solutions
# ============================================
print("\n📥 [2/3] Downloading MATH dataset...")
try:
    math_ds = load_dataset("hendrycks/competition_math")
    
    print(f"✅ MATH loaded!")
    print(f"   Train: {len(math_ds['train']):,} examples")
    print(f"   Test:  {len(math_ds['test']):,} examples")
    
    # Show example
    print("\n" + "-"*60)
    print("📝 EXAMPLE FROM MATH:")
    print("-"*60)
    example = math_ds['train'][0]
    print(f"PROBLEM:\n{example['problem']}\n")
    print(f"LEVEL: {example['level']}")
    print(f"TYPE: {example['type']}")
    print(f"SOLUTION:\n{example['solution'][:500]}...")
    print("-"*60)
    
    # Save to disk
    math_ds.save_to_disk("data/raw/math")
    print(f"💾 Saved to data/raw/math")
    
except Exception as e:
    print(f"❌ Error loading MATH: {e}")

# ============================================
# DATASET 3: Orca-Math (200K synthetic)
# Large dataset - optional but valuable
# ============================================
print("\n📥 [3/3] Downloading Orca-Math (this may take 2-3 minutes)...")
try:
    orca = load_dataset("microsoft/orca-math-word-problems-200k")
    
    print(f"✅ Orca-Math loaded!")
    print(f"   Train: {len(orca['train']):,} examples")
    
    # Show example
    print("\n" + "-"*60)
    print("📝 EXAMPLE FROM ORCA-MATH:")
    print("-"*60)
    example = orca['train'][0]
    print(f"QUESTION:\n{example['question']}\n")
    print(f"ANSWER:\n{example['answer'][:500]}...")
    print("-"*60)
    
    # Save to disk
    orca.save_to_disk("data/raw/orca_math")
    print(f"💾 Saved to data/raw/orca_math")
    
except Exception as e:
    print(f"⚠️ Orca-Math issue: {e}")
    print("   (Optional dataset - GSM8K and MATH are sufficient)")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("✅ STEP 1 COMPLETE: DATASETS DOWNLOADED")
print("="*60)
print("""
Downloaded to data/raw/:
├── gsm8k/       (~8.8K grade-school math problems)
├── math/        (~12.5K competition-level problems)
└── orca_math/   (~200K synthetic problems)

Total: ~220,000 problems with reasoning - ALL FREE!

Next: Run step2_format_data.py to convert to competition format
""")