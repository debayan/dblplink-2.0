import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import torch.nn.functional as F

class SimpleEntityLinker:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.device = device

    def format_input(self, mention, context, entity_name, entity_info_lines):
        entity_info_text = "\n".join(entity_info_lines)
        return f"""You are an assistant linking mentions to entities.
                   Document: {context}

                   Mention: {mention}

                   Candidate Entity: {entity_name}

                   Entity Info: {entity_info_text}

                   Question: Does the mention belong to this entity? Answer Yes/No.

                   Answer:"""

    def compute_yes_score(self, input_text):
        full_input = input_text + " Yes"
        inputs = self.tokenizer(full_input, return_tensors='pt', truncation=True, max_length=2048).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Log probability of " Yes" token after 'Answer:'
        log_probs = F.log_softmax(logits, dim=-1)
        yes_token_id = self.tokenizer(" Yes")["input_ids"][0]
        # Second last token is " Yes" because last token is <eos>
        yes_logprob = log_probs[0, -2, yes_token_id].item()

        return yes_logprob

    def score_entity(self, mention, context, entity_name, entity_info_lines):
        best_score = float("-inf")
        best_line = ""
        for line in entity_info_lines:
            prompt = self.format_input(mention, context, entity_name, line)
            score = self.compute_yes_score(prompt)
            if score > best_score:
                best_score = score
                best_line = line
        return best_score, best_line

    def rank_entities(self, mention, context, candidate_entities):
        ranked = []
        for entity_name, entity_info_lines in candidate_entities.items():
            score, best_line = self.score_entity(mention, context, entity_name, entity_info_lines)
            ranked.append((entity_name, score, best_line))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

# --------------------------
# 🧪 Example Usage
# --------------------------

if __name__ == "__main__":
    linker = SimpleEntityLinker("Qwen/Qwen2.5-0.5B-Instruct")

    mention = "Apple"
    context = "Steve Jobs founded Apple in his garage. It later became one of the most valuable companies in the world."

    candidate_entities = {
        "Apple Inc.": [
            "Apple Inc. is a technology company headquartered in Cupertino, California.",
            "Steve Jobs was a co-founder of Apple Inc.",
            "Apple Inc. develops iPhones, iPads, and Macs."
        ],
        "Apple (fruit)": [
            "Apples are a type of fruit that grow on trees.",
            "The apple is typically red, green, or yellow.",
            "Apple (fruit) is commonly found in temperate climates."
        ]
    }

    results = linker.rank_entities(mention, context, candidate_entities)

    for name, score, line in results:
        print(f"\nEntity: {name}")
        print(f"Score: {score:.4f}")
        print(f"Most Relevant Info Line: {line}")
