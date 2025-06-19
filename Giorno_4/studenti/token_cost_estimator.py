import tiktoken

# Prezzi aggiornati (giugno 2024) - input token
MODEL_PRICING = {
    "gpt-4o": 0.005,
    "gpt-4": 0.03,
    "gpt-3.5-turbo": 0.001,
}

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def estimate_cost(text: str, model: str = "gpt-4o") -> float:
    if model not in MODEL_PRICING:
        raise ValueError(f"Modello non supportato: {model}")
    
    num_tokens = count_tokens(text, model)
    price_per_1k = MODEL_PRICING[model]
    cost = (num_tokens / 1000) * price_per_1k
    return round(num_tokens, 2), round(cost, 6)

if __name__ == "__main__":
    print("Inserisci il testo da analizzare:")
    text_input = input("> ")
    print("Scegli il modello (gpt-4o, gpt-4, gpt-3.5-turbo):")
    model_choice = input("> ").strip()

    try:
        tokens, price = estimate_cost(text_input, model_choice)
        print(f"\nToken totali: {tokens}")
        print(f"Costo stimato: ${price}\n")
    except Exception as e:
        print(f"Errore: {e}")
