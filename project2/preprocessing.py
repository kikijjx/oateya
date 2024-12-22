import pandas as pd
from transformers import T5Tokenizer
import json

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df = df.dropna()
    df['english'] = df['english'].apply(lambda x: str(x).strip(' "'))
    df['russian'] = df['russian'].apply(lambda x: str(x).strip(' "'))
    
    tokenized_data = []
    for _, row in df.iterrows():
        english_tokens = tokenizer.encode(row['english'], add_special_tokens=True)
        russian_tokens = tokenizer.encode(row['russian'], add_special_tokens=True)
        tokenized_data.append({
            "english": english_tokens,
            "russian": russian_tokens
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tokenized_data, f, ensure_ascii=False, indent=4)

    print(f"Data preprocessed and saved to {output_file}")

if __name__ == "__main__":
    preprocess_data("sentences.csv", "tokenized_data.json")
