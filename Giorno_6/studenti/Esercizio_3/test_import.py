# check_azurerag.py
import ast
import os

print("=== ANALISI CONTENUTO AzureRag.py ===\n")

# Leggi il file
try:
    with open('AzureRag.py', 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"✓ File letto, dimensione: {len(content)} caratteri\n")
except Exception as e:
    print(f"✗ Errore lettura file: {e}")
    exit(1)

# 1. Cerca la classe
print("1. RICERCA CLASSE AzureRAGSystem:")
if 'class AzureRAGSystem' in content:
    print("   ✓ Stringa 'class AzureRAGSystem' trovata")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'class AzureRAGSystem' in line:
            print(f"   Linea {i+1}: {line}")
            # Mostra contesto
            for j in range(max(0, i-3), min(len(lines), i+5)):
                print(f"   {j+1}: {lines[j]}")
            break
else:
    print("   ✗ Classe AzureRAGSystem NON TROVATA!")
    
    # Cerca altre classi
    print("\n   Classi trovate nel file:")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('class '):
            print(f"   Linea {i+1}: {line.strip()}")

# 2. Verifica sintassi
print("\n2. VERIFICA SINTASSI:")
try:
    tree = ast.parse(content)
    print("   ✓ Sintassi Python valida")
    
    # Elenca tutte le classi
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    print(f"   Classi definite: {classes}")
    
    # Elenca funzioni top-level
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    print(f"   Funzioni top-level: {functions[:10]}...")
    
except SyntaxError as e:
    print(f"   ✗ ERRORE DI SINTASSI!")
    print(f"   Linea {e.lineno}: {e.msg}")
    print(f"   {e.text}")
    print(f"   {' ' * (e.offset-1)}^")

# 3. Test import diretto
print("\n3. TEST IMPORT:")
try:
    import AzureRag
    print("   ✓ Import modulo riuscito")
    attrs = [x for x in dir(AzureRag) if not x.startswith('_')]
    print(f"   Attributi pubblici: {attrs}")
    
    # Verifica se c'è un nome simile
    for attr in attrs:
        if 'azure' in attr.lower() or 'rag' in attr.lower():
            print(f"   Possibile match: {attr}")
            
except Exception as e:
    print(f"   ✗ Errore import: {e}")

# 4. Possibili problemi comuni
print("\n4. CONTROLLI AGGIUNTIVI:")

# Controlla indentazione mista
if '\t' in content and '    ' in content:
    print("   ⚠️ ATTENZIONE: Mix di tab e spazi nell'indentazione!")

# Controlla se la classe è dentro if __name__ == '__main__'
if 'if __name__' in content:
    # Trova dove
    main_block_line = None
    for i, line in enumerate(content.split('\n')):
        if 'if __name__' in line:
            main_block_line = i
            break
    
    if main_block_line and 'class AzureRAGSystem' in content:
        # Verifica se la classe è dopo
        class_line = None
        for i, line in enumerate(content.split('\n')):
            if 'class AzureRAGSystem' in line:
                class_line = i
                break
        
        if class_line and class_line > main_block_line:
            print("   ⚠️ PROBLEMA: La classe è definita dentro if __name__ == '__main__'!")