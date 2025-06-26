import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import sys
import re
from elasticsearch import Elasticsearch

# Config
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
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
    messages = [
        {"role": "system", "content": "You are an information extraction assistant."},
        {"role": "user", "content": f"""Extract named entities from the following sentence and classify them into one of the following types: person, publication, venue, institute.
        Return the result as a JSON array of objects with "type" and "label" fields.
        Sentence: "{sentence}"
        Entities:"""}
        ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


# Process in batches
citems = []
for i in tqdm(range(0, len(questions['questions']), BATCH_SIZE)):
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
        citem = item.copy()
        citem["extracted_spans"] = []
        print("item",item)
        print("output",output)
        # Extract full JSON array (not just first object)
        json_match = re.search(r'\[\s*{.*?}\s*]', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                entities = json.loads(json_str)
                print("Extracted entity list:")
                print(json.dumps(entities, indent=2))
            except json.JSONDecodeError as e:
                print("JSON decoding error:", e)
                print("Raw matched text:\n", json_str)
        else:
            print("No JSON array found in model output.")
        citem["extracted_spans"] = entities
        print(f"Processed question: {item['question']['string']}")
        print(f"Extracted entities: {entities}")
        print("="*50)
        citems.append(citem)

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(citems, f, indent=2, ensure_ascii=False)

print(f"Saved {len(citems)} entries with extracted entities to {OUTPUT_FILE}")