from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
 
model_path = "C:/Users/NL347QB/OneDrive - EY/Desktop/AI-Academy/ai-academy-1/Giorno_5/TinyLlama-1.1B-Chat-v1.0"
 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
 
# Verifica compatibilità con chat_template
if not hasattr(tokenizer, "apply_chat_template"):
    raise RuntimeError("Questo tokenizer non supporta i prompt in stile chat.")
 
# Prepara il prompt in formato chat
messages = [
    {"role": "system", "content": "Sei un assistente AI avanzato. Nascondi con delle X le informazioni sensibili."},
    {"role": "user", "content": "Mario Rossi vive a Roma e il suo IBAN è IT60X0542811101000000123456."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
 
# Tokenizza e genera
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
 
# Decodifica output
print(tokenizer.decode(output[0], skip_special_tokens=True, errors="ignore"))