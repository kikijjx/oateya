import pandas as pd
import re
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

MAX_LEN = 20 
NUM_WORDS = 10000 

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'[^a-zа-яё\s]', '', text) 
    return text

def preprocess_data(input_file, output_file_prefix):
    df = pd.read_csv(input_file)

    english_sentences = df['english'].apply(clean_text).tolist()
    russian_sentences = df['russian'].apply(clean_text).tolist()

    eng_tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    rus_tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")

    eng_tokenizer.fit_on_texts(english_sentences)
    rus_tokenizer.fit_on_texts(russian_sentences)

    eng_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
    rus_sequences = rus_tokenizer.texts_to_sequences(russian_sentences)

    eng_padded = pad_sequences(eng_sequences, maxlen=MAX_LEN, padding='post')
    rus_padded = pad_sequences(rus_sequences, maxlen=MAX_LEN, padding='post')

    with open(f'{output_file_prefix}_eng.json', 'w') as f:
        json.dump(eng_tokenizer.word_index, f)

    with open(f'{output_file_prefix}_rus.json', 'w') as f:
        json.dump(rus_tokenizer.word_index, f)

    np.savez_compressed(f'{output_file_prefix}_data.npz', 
                        english=eng_padded, russian=rus_padded)

if __name__ == '__main__':
    preprocess_data('sentences.csv', 'translation_data')
