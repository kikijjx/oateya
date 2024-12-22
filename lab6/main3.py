from gensim.models import Word2Vec
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from gensim.models import Word2Vec

models = []
for i in range(1, 8):
    model = Word2Vec.load(f'word2vec_model_{i}.model')
    models.append(model)

words_to_compare = ['disneyland', 'paris', 'ride']

for word in words_to_compare:
    print(f"Сравнение похожих слов для '{word}':\n")
    for i, model in enumerate(models):
        similar_words = model.wv.most_similar(word)
        print(f"Похожие слова к '{word}' (модель {i+1}):")
        for similar_word, similarity in similar_words:
            print(f"{similar_word}: {similarity:.4f}")
        print("\n")


def visualize_word(model, word, num_words=10):
    word_vectors = []
    word_labels = []

    similar_words = model.wv.most_similar(word, topn=num_words)
    word_vectors.append(model.wv[word])
    word_labels.append(word)
    for similar_word, _ in similar_words:
        word_vectors.append(model.wv[similar_word])
        word_labels.append(similar_word)

    word_vectors = np.array(word_vectors)

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    word_vectors_2d = tsne.fit_transform(word_vectors)

    return word_vectors_2d, word_labels

plt.figure(figsize=(18, 6))

for i, word in enumerate(words_to_compare):
    word_vectors_2d, word_labels = visualize_word(models[0], word)

    plt.subplot(1, 3, i + 1)
    for j, label in enumerate(word_labels):
        x, y = word_vectors_2d[j]
        plt.scatter(x, y, marker='o')
        plt.annotate(label, (x, y), fontsize=10)

    plt.title(word)

plt.tight_layout()
plt.show()