import re
import spacy
import pymorphy3
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict, Counter

nlp = spacy.load('ru_core_news_sm')
morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return re.split(r'[.!?;]+', text)

def lemmatize_and_filter(tokens):
    lemmatized_tokens = []
    for token in tokens:
        if token.pos_ not in ['AUX', 'DET', 'ADP', 'PUNCT', 'SPACE', 'CCONJ', 'SCONJ', 'SYM']:
            lemma = morph.parse(token.text)[0].normal_form
            lemmatized_tokens.append(lemma)
    return lemmatized_tokens

#filenames = ["iphone723.txt"]
filenames = ["гель929.txt"]
#filenames = ["роботпылесос4665.txt"]
#filenames = ["iphone723.txt", "гель929.txt", "роботпылесос4665.txt"]

all_texts = ""
for filename in filenames:
    with open(filename, "r", encoding="utf-8") as f:
        all_texts += f.read() + " "

sentences = preprocess_text(all_texts)
word_count = Counter()
word_pairs = defaultdict(int)

for sentence in sentences:
    tokens = nlp(sentence.strip())
    lemmatized_sentence = lemmatize_and_filter(tokens)
    
    for i, word in enumerate(lemmatized_sentence):
        word_count[word] += 1
        for j in range(i + 1, len(lemmatized_sentence)):
            pair = tuple(sorted([word, lemmatized_sentence[j]]))
            word_pairs[pair] += 1

top = [word for word, _ in word_count.most_common(30)]

top_G = nx.Graph()
for word in top:
    top_G.add_node(word, size=word_count[word])

filtered_word_pairs = {pair: count for pair, count in word_pairs.items() if pair[0] in top and pair[1] in top}
for pair, count in filtered_word_pairs.items():
    top_G.add_edge(pair[0], pair[1], weight=count)

pos = nx.spring_layout(top_G)
node_x, node_y, node_sizes = [], [], []

for node in top_G.nodes:
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_sizes.append(top_G.nodes[node]['size'] * 0.5)

edge_x, edge_y = [], []
for edge in top_G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(x=edge_x, y=edge_y)
node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                         marker=dict(showscale=True, size=node_sizes, line_width=2))
node_trace.text = [f'{node}: {word_count[node]}' for node in top_G.nodes]

fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=filenames[0], titlefont_size=16, hovermode='closest'))

fig.show()
