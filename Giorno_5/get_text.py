import pathlib

def get_text(file_name):
    with open(pathlib.Path(__file__).parent / file_name, "r", encoding="utf-8") as file:
        text = file.read()
        return text.strip() if text else "No content found in the file."
    
if __name__ == "__main__":
    print(get_text("cleaned_output.txt"))