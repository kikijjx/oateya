import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json

MAX_LEN = 20

def load_tokenizers(eng_file, rus_file):
    with open(eng_file, 'r') as f:
        eng_word_index = json.load(f)
    with open(rus_file, 'r') as f:
        rus_word_index = json.load(f)
    
    eng_tokenizer = Tokenizer()
    eng_tokenizer.word_index = eng_word_index
    
    rus_tokenizer = Tokenizer()
    rus_tokenizer.word_index = rus_word_index
    
    return eng_tokenizer, rus_tokenizer

def translate(sentence, model, tokenizer_input, tokenizer_output):
    sequence = tokenizer_input.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

    prediction = model.predict([sequence, sequence])

    output_sentence = ''
    for word_id in np.argmax(prediction[0], axis=1):
        word = [k for k, v in tokenizer_output.word_index.items() if v == word_id]
        if word:
            output_sentence += word[0] + ' '
    
    return output_sentence.strip()

if __name__ == '__main__':
    model = load_model('eng_rus_seq2seq.h5')
    eng_tokenizer, rus_tokenizer = load_tokenizers('translation_data_eng.json', 'translation_data_rus.json')

    sentence = input("Введите предложение для перевода: ")
    print("Перевод:", translate(sentence, model, eng_tokenizer, rus_tokenizer))
