from G5E1_thomson import process_document_and_ask

risposta = process_document_and_ask(
    document_path="documento_test.txt",
    system_prompt="Sei un assistente aziendale.",
    instruction_prompt="""
    Fornisci un riassunto e suggerisci eventuali risposte adeguate.
    """,
    max_completion_tokens=1200,
    temperature=1.0
)

print(risposta)

