from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "C:/DesktopNoOneDrive/ai-academy/Giorno_5/tinyllama_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Verifica compatibilità con chat_template
if not hasattr(tokenizer, "apply_chat_template"):
    raise RuntimeError("Questo tokenizer non supporta i prompt in stile chat.")

# Prepara il prompt in formato chat
messages = [
    {"role": "system", "content": "Sei un assistente intelligente."},
    {"role": "user", "content": "Qual è la capitale dell'Italia?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenizza e genera
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

# Decodifica output
print(tokenizer.decode(output[0], skip_special_tokens=True, errors="ignore"))
