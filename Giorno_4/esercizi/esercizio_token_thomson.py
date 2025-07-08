import tiktoken

# Prezzo aggiornato GPT-4.1 input
Prezzo_per_token = 2.00 / 1_000_000

def calcola_gpt4_input_cost(prompt: str) -> tuple[int, float]:
    """
    Calcola il numero di token e il costo dell'input per GPT-4.1.

    Argomenti:
        prompt (str): Testo di input.

    Ritorna:
        (num_tokens, cost_usd)
    """
    enc = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(enc.encode(prompt))
    costo_usd = num_tokens * Prezzo_per_token
    return num_tokens, costo_usd

if __name__ == "__main__":
    prompt = input("Inserisci il prompt: ")
    tokens, cost = calcola_gpt4_input_cost(prompt)
    print(f"\nNumero di token: {tokens}")
    print(f"Prezzo input: ${cost:.6f}")