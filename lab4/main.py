import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


data = pd.read_csv('tmdb_5000_movies.csv')
movies = data[['original_title', 'overview']].dropna(subset=['overview'])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

movies['cleaned_overview'] = movies['overview'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['cleaned_overview'])
def plott():
    tsne = TSNE(random_state=0, verbose=True)
    tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
    
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.show()

cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix).flatten()
eucl_sim = euclidean_distances(tfidf_matrix[0], tfidf_matrix).flatten()
cos_res = cos_sim.argsort()[::-1][:5]
print(f"похожие фильмы для '{movies.iloc[0]['original_title']}':")
for idx in cos_res:
    print(f"{movies.iloc[idx]['original_title']} /// {cos_sim[idx]}")
eucl_res = eucl_sim.argsort()[:5]
print(f"\nпохожие фильмы для '{movies.iloc[0]['original_title']}' (Евклидово расстояние):")
for idx in eucl_res:
    print(f"{movies.iloc[idx]['original_title']} /// {eucl_sim[idx]}")

plott()

newm = "An author returns to his hometown of Jerusalem's Lot in search of inspiration for his next book, only to discover that the townspeople are being attacked by a bloodthirsty vampire.."
cleaned = preprocess_text(newm)
newvec = vectorizer.transform([cleaned])

cos_sim = cosine_similarity(newvec, tfidf_matrix).flatten()
eucl_sim = euclidean_distances(newvec, tfidf_matrix).flatten()
print(f'\nописание нового фильма: {newm}')
cos_res = cos_sim.argsort()[::-1][:5]
print("\nпохожие фильмы для нового фильма:")
for idx in cos_res:
    print(f"{movies.iloc[idx]['original_title']} /// {cos_sim[idx]}")


eucl_res = eucl_sim.argsort()[:5]
print("\nпохожие фильмы для нового фильма (Евклидово расстояние):")
for idx in eucl_res:
    print(f"{movies.iloc[idx]['original_title']} /// {eucl_sim[idx]}")
    #print(f"{movies.iloc[idx]['overview']} /// {eucl_sim[idx]}")




