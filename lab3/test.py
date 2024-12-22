import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import spacy
from nltk.stem import SnowballStemmer

data = pd.read_csv('youtoxic_english_1000.csv', sep=',')

category = 'IsRacist'
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def process_text(text, lemmatizer_func=None, stemmer_func=None):
    tokens = word_tokenize(text)
    if lemmatizer_func:
        res2 = lemmatizer_func(tokens)
    else:
        res2 = tokens

    if stemmer_func:
        res = stemmer_func(res2)
    else:
        res = res2

    return res

def make_combined_plots(toxic_freq, non_toxic_freq, title, num_words=20, block=False):
    most_common_toxic = toxic_freq.most_common(num_words)
    most_common_non_toxic = non_toxic_freq.most_common(num_words)

    toxic_words, toxic_frequencies = zip(*most_common_toxic)
    non_toxic_words, non_toxic_frequencies = zip(*most_common_non_toxic)

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.barh(toxic_words, toxic_frequencies)
    plt.xlabel('Частота')
    plt.title(f'{title}')
    plt.gca().invert_yaxis()

    plt.subplot(2, 2, 2)
    toxic_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(toxic_freq))
    plt.imshow(toxic_wordcloud, interpolation='bilinear')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.barh(non_toxic_words, non_toxic_frequencies)
    plt.xlabel('Частота')
    plt.title(f'{title} (остальные)')
    plt.gca().invert_yaxis()

    plt.subplot(2, 2, 4)
    non_toxic_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(non_toxic_freq))
    plt.imshow(non_toxic_wordcloud, interpolation='bilinear')
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=block)

def word_frequency(processed_texts):
    words = [word for text in processed_texts for word in text]
    return Counter(words)

def nltk_lem(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def spacy_lem(tokens):
    return [token.lemma_ for token in nlp(' '.join(tokens))]

def porter_stem(tokens):
    return [porter_stemmer.stem(token) for token in tokens]

def snowball_stem(tokens):
    return [snowball_stemmer.stem(token) for token in tokens]

def inpf():
    toxic_texts = data[data[category] == 1]['cleaned_text']
    non_toxic_texts = data[data[category] == 0]['cleaned_text']

    toxic_freq_nltk = word_frequency(toxic_texts.apply(
        lambda text: process_text(text, lemmatizer_func=nltk_lem)))
    non_toxic_freq_nltk = word_frequency(non_toxic_texts.apply(
        lambda text: process_text(text, lemmatizer_func=nltk_lem)))

    make_combined_plots(toxic_freq_nltk, non_toxic_freq_nltk, f'NLTK Lemmatization')

    toxic_freq_spacy = word_frequency(toxic_texts.apply(
        lambda text: process_text(text, lemmatizer_func=spacy_lem)))
    non_toxic_freq_spacy = word_frequency(non_toxic_texts.apply(
        lambda text: process_text(text, lemmatizer_func=spacy_lem)))

    make_combined_plots(toxic_freq_spacy, non_toxic_freq_spacy, f'spaCy Lemmatization')

    toxic_freq_porter = word_frequency(toxic_texts.apply(
        lambda text: process_text(text, stemmer_func=porter_stem)))
    non_toxic_freq_porter = word_frequency(non_toxic_texts.apply(
        lambda text: process_text(text, stemmer_func=porter_stem)))

    make_combined_plots(toxic_freq_porter, non_toxic_freq_porter, f'Porter Stemming')

    toxic_freq_snowball = word_frequency(toxic_texts.apply(
        lambda text: process_text(text, stemmer_func=snowball_stem)))
    non_toxic_freq_snowball = word_frequency(non_toxic_texts.apply(
        lambda text: process_text(text, stemmer_func=snowball_stem)))

    make_combined_plots(toxic_freq_snowball, non_toxic_freq_snowball, f'Snowball Stemming', block=True)


data['cleaned_text'] = data['Text'].apply(clean_text)
inpf()
