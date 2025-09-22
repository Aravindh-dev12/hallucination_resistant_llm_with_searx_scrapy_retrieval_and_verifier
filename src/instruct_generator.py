from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import load_config

cfg = load_config()
MODEL = cfg['model']['generator_instruct']

class InstructGenerator:
    def __init__(self, model_name=None, device=None, low_memory=False):
        self.model_name = model_name or MODEL
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        try:
            load_kwargs = {}
            if self.device.startswith('cuda') and low_memory:
                load_kwargs.update({'device_map':'auto', 'load_in_8bit':True})
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs).to(self.device)
        except Exception as e:
            print('Load warning:', e)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
    def promptify(self, instruction, context=''):
        return f"""You are an assistant. Use the context to answer.\n\nContext:\n{context}\n\nInstruction: {instruction}\n\nResponse:"""
    def generate(self, instruction, context='', max_tokens=256, temperature=0.0):
        prompt = self.promptify(instruction, context)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=(temperature>0.0), temperature=temperature, pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split('Response:')[-1].strip()
