import numpy as np
from scipy.spatial.distance import cosine
import nltk
from nltk.tokenize import word_tokenize

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings = load_glove_embeddings('../word_vec/glove.6B.100d.txt')

def most_similar_word(word, tech_words, embeddings_index):
    if word not in embeddings_index:
        return None

    word_vec = embeddings_index[word]
    best_similarity = float('inf')
    best_word = None

    for tech_word in tech_words:
        if tech_word in embeddings_index:
            tech_word_vec = embeddings_index[tech_word]
            similarity = cosine(word_vec, tech_word_vec)
            if similarity < best_similarity:
                best_similarity = similarity
                best_word = tech_word

    return best_word

def chatbot_response(user_input):
    tokens = word_tokenize(user_input.lower())
    tech_words_detected = []

    for token in tokens:
        similar_tech_word = most_similar_word(token, ferramentas_linguagens, glove_embeddings)
        if similar_tech_word:
            tech_words_detected.append(similar_tech_word)

    if tech_words_detected:
        return f"Você mencionou palavras relacionadas à tecnologia: {', '.join(set(tech_words_detected))}."
    else:
        return "Nenhuma palavra relacionada à tecnologia foi detectada."

nltk.download('punkt')

ferramentas_linguagens = set([
    # Linguagens de Programação
    'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
    'go', 'typescript', 'kotlin', 'r', 'scala', 'perl', 'html', 'css', 'sql',
    
    # Ferramentas de Desenvolvimento
    'git', 'github', 'gitlab', 'docker', 'kubernetes', 'jenkins', 'nginx', 
    'vscode', 'eclipse', 'intellij', 'xcode', 'atom', 'webpack', 'npm', 'yarn',
    'postman', 'selenium', 'trello', 'apache', 'mysql', 'postgresql', 'mongodb'
])

# Exemplo de uso
user_input = "Estou interessado em aprender mais sobre redes de computadores. Já trabalhei com python"
response = chatbot_response(user_input)
print(response)
