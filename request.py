import numpy as np
from scipy.spatial.distance import cosine
import nltk
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify

# Inicializa o Flask
app = Flask(__name__)

# Carrega os embeddings GloVe
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings = load_glove_embeddings('word_vec/glove.6B.100d.txt')

def most_similar_word(word, tech_words, embeddings_index):
    if word not in embeddings_index:
        return None, float('inf')

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

    return best_word, best_similarity

def chatbot_response(user_input):
    tokens = word_tokenize(user_input.lower())
    tech_words_detected = {}

    # Define um limite para a similaridade (quanto menor, mais relevante)
    similarity_threshold = 0.5

    for token in tokens:
        similar_tech_word, similarity = most_similar_word(token, ferramentas_linguagens, glove_embeddings)
        if similar_tech_word and similarity <= similarity_threshold:
            # Se a palavra já foi detectada, mantemos a de menor similaridade
            if (similar_tech_word not in tech_words_detected or 
                    similarity < tech_words_detected[similar_tech_word]):
                tech_words_detected[similar_tech_word] = similarity

    # Agora, retornamos o dicionário
    return tech_words_detected

def format_response(tech_words_detected, habilidades_necessarias):
    habilidades_descritas = []
    habilidades_relacionadas = []
    habilidades_faltantes = list(habilidades_necessarias)

    for word, similarity in tech_words_detected.items():
        if similarity == 0:
            habilidades_descritas.append(word)
            # Remove das habilidades faltantes se foi descrita
            if word in habilidades_faltantes:
                habilidades_faltantes.remove(word)
        else:
            habilidades_relacionadas.append(f"{word} tem a similaridade com o texto de {round(similarity, 3)} (quanto mais próximo de 0 mais relacionado com o texto)")
    
    return {
        'habilidades_descritas': habilidades_descritas,
        'habilidades_relacionadas': habilidades_relacionadas,
        'habilidades_faltantes': habilidades_faltantes
    }

nltk.download('punkt')

ferramentas_linguagens = set([
    # Linguagens de Programação
    'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
    'typescript', 'kotlin', 'scala', 'perl', 'html', 'css', 'sql', 'node.js',
    'react', 'reactnative',
    
    # Ferramentas de Desenvolvimento
    'git', 'github', 'gitlab', 'docker', 'kubernetes', 'jenkins', 'nginx', 
    'vscode', 'eclipse', 'intellij', 'xcode', 'atom', 'webpack', 'npm', 'yarn',
    'postman', 'selenium', 'trello', 'apache', 'mysql', 'postgresql', 'mongodb'
])

# Dicionário de grupos com habilidades necessárias
grupos_habilidades = {
    "desenvolvedor web": {'html', 'css', 'javascript', 'react', 'node.js', 'git', 'docker', 'mysql'},
    "desenvolvedor mobile": {'java', 'kotlin', 'swift', 'reactnative', 'git', 'docker', 'sqlite'}
}

# Endpoint para receber a request com JSON e retornar a resposta
@app.route('/analyze', methods=['POST'])
def analyze_message():
    data = request.get_json()

    # Verifica se o JSON contém os campos 'message' e 'grupo'
    if 'message' not in data or 'grupo' not in data:
        return jsonify({'error': 'Os campos "message" e "grupo" são obrigatórios.'}), 400

    user_input = data['message']
    grupo = data['grupo']

    # Verifica se o grupo informado existe no dicionário
    if grupo not in grupos_habilidades:
        return jsonify({'error': f'Grupo "{grupo}" não encontrado. Grupos disponíveis: {list(grupos_habilidades.keys())}'}), 400

    # Habilidades necessárias para o grupo
    habilidades_necessarias = grupos_habilidades[grupo]

    # Obtemos o dicionário de palavras detectadas e suas similaridades
    detected_words = chatbot_response(user_input)

    # Formata a resposta final, incluindo as habilidades faltantes
    formatted_response = format_response(detected_words, habilidades_necessarias)

    return jsonify(formatted_response)

# Inicia o servidor Flask
if __name__ == '__main__':
    app.run(debug=True)
