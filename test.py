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

def format_response(tech_words_detected):
    habilidades_descritas = []
    habilidades_relacionadas = []

    for word, similarity in tech_words_detected.items():
        if similarity == 0:
            habilidades_descritas.append(word)
        else:
            habilidades_relacionadas.append(f"{word} (similaridade: {round(similarity, 3)})")
    
    response = ""
    if habilidades_descritas:
        response += f"Habilidades descritas pelo colaborador: {', '.join(habilidades_descritas)}.\n"
    if habilidades_relacionadas:
        response += f"Habilidades relacionadas às descritas pelo colaborador: {', '.join(habilidades_relacionadas)}."

    return response

nltk.download('punkt')

ferramentas_linguagens = set([
    # Linguagens de Programação
    'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
    'typescript', 'kotlin', 'scala', 'perl', 'html', 'css', 'sql', 'node',
    'react', 'reactnative',
    
    # Ferramentas de Desenvolvimento
    'git', 'github', 'gitlab', 'docker', 'kubernetes', 'jenkins', 'nginx', 
    'vscode', 'eclipse', 'intellij', 'xcode', 'atom', 'webpack', 'npm', 'yarn',
    'postman', 'selenium', 'trello', 'apache', 'mysql', 'postgresql', 'mongodb'
])

# Exemplo de uso
user_input = '''
    Sou Pedro Almeida, recém-formado em Ciência da Computação pela Universidade de São Paulo, e estou entusiasmado com a oportunidade de integrar a equipe como desenvolvedor. Aqui estão algumas das minhas habilidades e experiências principais:
    Desenvolvimento Full-Stack: Trabalhei com React no front-end e Node.js com Express no back-end para criar um aplicativo de gestão de tarefas. Experiência com integração de APIs RESTful e boas práticas de arquitetura de software.
    Banco de Dados: Experiência com MySQL e MongoDB, incluindo modelagem de dados e otimização de consultas. Implementação de soluções de cache para melhorar a performance.
    Ferramentas e Práticas Modernas: Familiaridade com Git para controle de versão, Docker para containerização e CI/CD com GitHub Actions. Conhecimento em testes unitários e de integração.
    Aprendizado Contínuo: Atualmente, estou aprofundando meus conhecimentos em arquiteturas de microserviços e desenvolvimento ágil.
    Estou animado para trazer minha paixão por tecnologia e minha disposição para aprender para a equipe. Agradeço pela oportunidade e estou à disposição para discutir mais detalhes.
'''

# Obtemos o dicionário de palavras detectadas e suas similaridades
detected_words = chatbot_response(user_input)

# Formata a resposta final
formatted_response = format_response(detected_words)
print(formatted_response)
