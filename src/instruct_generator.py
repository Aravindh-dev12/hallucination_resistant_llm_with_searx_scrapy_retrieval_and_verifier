import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_config

cfg = load_config()
MODEL = cfg["model"]["generator_instruct"]


class InstructGenerator:
    def __init__(self, model_name=None, device=None, low_memory=False):
        self.model_name = model_name or MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
        ).to(self.device)
        self.model.eval()

    def _inputs(self, instruction, context):
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied evidence. Give a concise answer "
                    "of at most four sentences. Do not invent facts. If the evidence "
                    "does not answer the question, say that it is insufficient."
                ),
            },
            {
                "role": "user",
                "content": f"Evidence:\n{context}\n\nQuestion: {instruction}",
            },
        ]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                messages[0]["content"]
                + "\n\n"
                + messages[1]["content"]
                + "\n\nAnswer:"
            )
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1800,
        ).to(self.device)

    def generate(self, instruction, context="", max_tokens=256, temperature=0.0):
        inputs = self._inputs(instruction, context)
        generation = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation["temperature"] = temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
