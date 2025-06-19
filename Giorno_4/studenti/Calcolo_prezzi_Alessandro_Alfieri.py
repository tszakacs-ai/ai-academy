import tiktoken

prompt = "Quando è morto Napoleone Bonaparte?"

def calcola_prezzo(prompt):
    # 1. Tokenizzazione del prompt
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(prompt)
    
    # 2. Calcolo del prezzo
    prezzo_prompt = len(tokens) * (2.00 / 1000000)  # ipotizziamo 0.01 euro per token
    
    return prezzo_prompt

print(f"Prompt: {prompt}")
print(f"Il prezzo del prompt è: {calcola_prezzo(prompt):.6f} euro")