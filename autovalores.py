import numpy as np

# 1. Criando uma Matriz (representando a transformação de dados na IA)
# Vamos usar uma matriz simples 2x2
matriz_ia = np.array([[4, 2],
                      [1, 3]])

# 2. A Mágica Matemática: Extraindo Autovalores e Autovetores
autovalores, autovetores = np.linalg.eig(matriz_ia)

# 3. Exibindo os resultados
print("=== Análise da Matriz ===")
print(f"Os Autovalores encontrados foram: {autovalores}")
print(f"Os Autovetores correspondentes são:\n{autovetores}")