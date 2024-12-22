import re
import spacy
import pymorphy3
import networkx as nx
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations

# Загружаем модели
nlp = spacy.load("ru_core_news_sm")
morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text):
    # Убираем нежелательные символы и числа
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

def lemmatize_words(text):
    doc = nlp(text)
    lemmatized_words = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ not in ["DET", "ADP", "CCONJ", "SCONJ", "PART", "PRON"]:
                lemma = morph.parse(token.text)[0].normal_form
                lemmatized_words.append(lemma)
    return lemmatized_words

def build_graph(sentences):
    word_counts = Counter()  # Считаем частоту каждого слова
    pair_counts = Counter()  # Считаем частоту каждой пары слов

    # Считаем количество слов и пар слов
    for sent in sentences:
        words = lemmatize_words(sent)
        word_counts.update(words)
        pairs = combinations(words, 2)
        pair_counts.update(pairs)

    # Создаем граф
    G = nx.Graph()

    # Добавляем вершины и их веса (частоты)
    for word, count in word_counts.items():
        G.add_node(word, weight=count)

    # Добавляем рёбра и их веса (частоты)
    for (word1, word2), count in pair_counts.items():
        G.add_edge(word1, word2, weight=count)

    return G, word_counts, pair_counts

def plot_graph(G, word_counts):
    # Ограничиваем граф до 10 самых частых слов
    top_words = word_counts.most_common(10)
    top_words_set = set(word for word, _ in top_words)

    # Создаем новый граф с ограничением по словам
    G_top = G.subgraph(top_words_set)

    # Определяем положение узлов на графе
    pos = nx.spring_layout(G_top, seed=42)

    # Получаем веса для вершин и рёбер
    node_sizes = [G_top.nodes[n]['weight'] * 100 for n in G_top.nodes()]  # Размер узлов
    edge_weights = [G_top[u][v]['weight'] for u, v in G_top.edges()]  # Вес рёбер

    # Создаем график
    fig = go.Figure()

    # Добавляем узлы
    for node in G_top.nodes():
        fig.add_trace(go.Scatter(
            x=[pos[node][0]],
            y=[pos[node][1]],
            text=node,
            mode='markers+text',
            marker=dict(size=G_top.nodes[node]['weight'] * 10, color='skyblue', line=dict(color='black', width=1)),  # Упрощено
            textposition='bottom center'
        ))

    # Добавляем рёбра
    for edge in G_top.edges():
        fig.add_trace(go.Scatter(
            x=[pos[edge[0]][0], pos[edge[1]][0]],
            y=[pos[edge[0]][1], pos[edge[1]][1]],
            mode='lines',
            line=dict(width=G_top.edges[edge]['weight'], color='steelblue'),  # Упрощено
            hoverinfo='none'
        ))

    # Обновляем макет графика
    fig.update_layout(title="Граф слов с весами вершин и рёбер (Топ 10 слов)", showlegend=False, 
                      xaxis=dict(showgrid=False, zeroline=False), 
                      yaxis=dict(showgrid=False, zeroline=False),
                      margin=dict(l=40, r=40, t=40, b=40))  # Добавлены поля для удобства
    fig.show()

# Чтение и обработка текста
file_names = ['iphone723.txt']
sentences = []

for file in file_names:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    cleaned_text = preprocess_text(text)
    sentences.append(cleaned_text)

# Создание графа
G, word_counts, pair_counts = build_graph(sentences)

# Визуализация графа
plot_graph(G, word_counts)
