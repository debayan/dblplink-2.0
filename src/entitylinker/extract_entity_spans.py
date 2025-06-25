import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE = 16  # Qwen2.5-0.5B is small enough for small batches
INPUT_FILE = "dblp_quad/questions_valid.json"
OUTPUT_FILE = "dblp_quad/questions_valid_extracted_spans.json"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load input questions
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

# Template for prompt
def make_prompt(sentence):
    return f"""Extract named entities from the following sentence and classify them into one of the following types: person, publication, venue, institute.

Return the result as a JSON array of objects with "type" and "label" fields.

Sentence: "{sentence}"
Entities:"""

# Process in batches
all_results = []
for i in tqdm(range(0, len(questions), BATCH_SIZE)):
    batch = questions['questions'][i:i+BATCH_SIZE]
    prompts = [make_prompt(q['question']['string']) for q in batch]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for item, output in zip(batch, decoded_outputs):
        entity_json = output.split("Entities:", 1)[-1].strip()
        try:
            parsed = json.loads(entity_json)
        except json.JSONDecodeError:
            parsed = []
        item["extracted_spans"] = parsed
        print(f"Processed question: {item['question']['string']}")
        print(f"Extracted entities: {parsed}")
        

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data)} entries with extracted entities to {OUTPUT_FILE}")