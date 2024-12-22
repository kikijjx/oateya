import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping

# Параметры
EMBEDDING_DIM = 256
LATENT_DIM = 512
EPOCHS = 30
BATCH_SIZE = 128
MAX_LEN = 20
NUM_WORDS = 10000

def load_data(data_file):
    data = np.load(data_file)
    return data['english'], data['russian']

def build_seq2seq_model():
    encoder_inputs = Input(shape=(MAX_LEN,))
    decoder_inputs = Input(shape=(MAX_LEN,))

    encoder_embedding = Embedding(NUM_WORDS, EMBEDDING_DIM)(encoder_inputs)
    decoder_embedding = Embedding(NUM_WORDS, EMBEDDING_DIM)(decoder_inputs)

    encoder_lstm = LSTM(LATENT_DIM, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(NUM_WORDS, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

if __name__ == '__main__':
    english_data, russian_data = load_data('translation_data_data.npz')

    decoder_input_data = np.pad(russian_data[:, :-1], ((0, 0), (1, 0))) 
    decoder_target_data = russian_data

    model = build_seq2seq_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    early_stop = EarlyStopping(patience=3)
    model.fit([english_data, decoder_input_data], decoder_target_data,
              batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop])

    model.save('eng_rus_seq2seq.h5')
