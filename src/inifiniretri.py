import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import torch.nn.functional as F
import sys


class InfiniRetri:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", chunk_size=2048, topk_tokens=50, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, output_attentions=True).to(device)
        self.chunk_size = chunk_size
        self.topk = topk_tokens
        self.device = device
        self.cache_sentences = []

    def chunk_text(self, text):
        tokens = self.tokenizer.encode(text)
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i:i+self.chunk_size]

    def retrieve_sentences(self, input_ids, attention):
        attn = attention[-1]  # last layer
        attn_sum = attn.sum(dim=1).squeeze(0)
        token_relevance = attn_sum[-1]
        conv = F.conv1d(token_relevance.unsqueeze(0).unsqueeze(0),
                        torch.ones(1, 1, 3).to(self.device), padding=1).squeeze()
        topk_idxs = torch.topk(conv, self.topk).indices.cpu().tolist()
        tokens = input_ids.cpu().tolist()[0]
        text = self.tokenizer.decode(tokens)
        sents = text.split('. ')
        sent_bounds = []
        idx = 0
        for sent in sents:
            length = len(self.tokenizer.encode(sent + '.'))
            sent_bounds.append((idx, sent + '.'))
            idx += length
        selected = []
        for tid in topk_idxs:
            for start, sent in sent_bounds:
                if start <= tid < start + len(self.tokenizer.encode(sent)):
                    selected.append(sent)
                    break
        return list(dict.fromkeys(selected))

    def format_prompt(self, context, question):
        messages = [
            {"role": "system", "content": "You are Qwen. Please answer the question based on the context. Keep your answer short and precise."},
            {"role": "user", "content": f"Context: {context.strip()}\n\nQuestion: {question.strip()}"},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, text, question):
        self.cache_sentences = []
        all_chunks = list(self.chunk_text(text))
    
        # For each chunk, retrieve relevant sentences based only on that chunk's tokens
        for chunk in all_chunks:
            context = self.tokenizer.decode(chunk)
            inputs_chunk = self.tokenizer(context, return_tensors='pt', truncation=True, max_length=4096).to(self.device)
            with torch.no_grad():
                outputs_chunk = self.model(**inputs_chunk)
            attn = outputs_chunk.attentions
            new_sents = self.retrieve_sentences(inputs_chunk.input_ids, attn)
            self.cache_sentences.extend(new_sents)
    
        # After collecting relevant context sentences, format the prompt for generation
        final_prompt = self.format_prompt(' '.join(self.cache_sentences), question)
        #print("final_prompt:", final_prompt)
    
        final_input = self.tokenizer(final_prompt, return_tensors='pt', truncation=True, max_length=4096).to(self.device)
    
        generation_config = GenerationConfig(
            max_new_tokens=100,
            do_sample=False,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    
        output_ids = self.model.generate(**final_input, generation_config=generation_config)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        #print("decoded:", decoded)
        try:
            answer = decoded.strip().split("assistant\n")[-1]
        except Exception as err:
            print(err, "malformed response")
            answer = ""
        return answer


def load_hotpotqa_samples(path, max_samples=10):
    with open(path) as f:
        data = json.load(f)
    examples = []
    for item in data:
        context = ' '.join([' '.join(para[1]) for para in item['context']])
        question = item['question']
        answer = item['answer']
        examples.append((context, question, answer))
    return random.sample(examples, min(len(examples), max_samples))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ir = InfiniRetri(model_name="Qwen/Qwen2.5-0.5B-Instruct", chunk_size=1024, topk_tokens=30, device=device)

    hotpot_path = "hotpot_train_v1.1.json"  # Replace with actual path
    samples = load_hotpotqa_samples(hotpot_path, max_samples=100)

    for context, question, gold_answer in samples:
        print(f"Q: {question}")
        try:
            predicted = ir.generate(context, question)
        except Exception as err:
            print(err, "skipping ...")
            continue
            
        print(f"Predicted: {predicted}")
        print(f"Gold: {gold_answer}")
        print("=" * 80)

