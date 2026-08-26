import numpy as np

# 1. Simulando Embeddings (vetores)
# apenas estudo. esses números são gerados por modelos (como o Word2Vec ou da OpenAI etc)
# criando vetores fictícios de 3 dimensões para ilustrar conceito.
vetor_cachorro = np.array([0.9, 0.8, 0.1])
vetor_gato = np.array([0.8, 0.9, 0.2])
vetor_carro = np.array([0.1, 0.1, 0.9])

# 2. função matemática para calcular o Produto Escalar (dot product)
def calcular_similaridade(vetor_a, vetor_b):
    # o numPy faz a multiplicação elemento por elemento e soma tudo nativamente
    produto_escalar = np.dot(vetor_a, vetor_b)
    return produto_escalar

# 3. Executando testes de similaridade
sim_cachorro_gato = calcular_similaridade(vetor_cachorro, vetor_gato)
sim_cachorro_carro = calcular_similaridade(vetor_cachorro, vetor_carro)

# 4. exibindo os resultados
print(f"Similaridade entre Cachorro e Gato: {sim_cachorro_gato:.2f}")
print(f"Similaridade entre Cachorro e Carro: {sim_cachorro_carro:.2f}")