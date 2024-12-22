import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from gensim.models import Word2Vec

df = pd.read_csv('DisneylandReviews.csv', encoding='cp1251')

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

tqdm.pandas()
df['Review_Text'] = df['Review_Text'].progress_apply(preprocess_text)

def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

df['Sentences'] = df['Review_Text'].progress_apply(split_into_sentences)

def sentences_to_words(sentences):
    return [word_tokenize(sentence) for sentence in sentences]

df['Sentences'] = df['Sentences'].progress_apply(sentences_to_words)

all_sentences = [sentence for sublist in df['Sentences'] for sentence in sublist]








models = [
    Word2Vec(sentences=all_sentences, vector_size=150, window=7, min_count=3, sg=0),
    Word2Vec(sentences=all_sentences, vector_size=50, window=7, min_count=3, sg=0),
    Word2Vec(sentences=all_sentences, vector_size=150, window=2, min_count=3, sg=0),
    Word2Vec(sentences=all_sentences, vector_size=150, window=7, min_count=1, sg=0),
    Word2Vec(sentences=all_sentences, vector_size=150, window=7, min_count=3, sg=1),
    Word2Vec(sentences=all_sentences, vector_size=50, window=7, min_count=3, sg=1),
    Word2Vec(sentences=all_sentences, vector_size=150, window=2, min_count=3, sg=1),
    Word2Vec(sentences=all_sentences, vector_size=150, window=7, min_count=1, sg=1)
]







for i, model in enumerate(models):
    model.save(f'word2vec_model_{i+1}.model')
    print(f'Model {i+1} saved')

print('Всё')
