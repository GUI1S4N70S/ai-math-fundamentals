from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Nosso Dataset (Os dados que a IA vai usar para aprender)
mensagens = [
    "ganhe dinheiro rápido, clique aqui agora",        # Spam
    "oferta imperdível, produto grátis",               # Spam
    "bom dia equipe, a reunião de amanhã está confirmada?", # Normal
    "segue em anexo o relatório mensal de vendas",     # Normal
    "você ganhou um prêmio, resgate seu pix grátis"    # Spam
]

# Labels: 1 = Spam, 0 = Mensagem Normal
labels = [1, 1, 0, 0, 1]

# 2. Transformando Texto em Matemática (Matrizes!)
vetorizador = CountVectorizer()
matriz_palavras = vetorizador.fit_transform(mensagens)

# 3. O Motor Estatístico (Teorema de Bayes na Prática)
# A IA está aprendendo a probabilidade de cada palavra pertencer a um Spam
modelo_bayes = MultinomialNB()
modelo_bayes.fit(matriz_palavras, labels)

# 4. Inferência: Testando a IA com mensagens INÉDITAS
novas_mensagens = [
    "urgente, libere seu acesso grátis agora",
    "bom dia, podemos reagendar a reunião para sexta?"
]
novas_matrizes = vetorizador.transform(novas_mensagens)

# A IA calcula a probabilidade e decide
previsoes = modelo_bayes.predict(novas_matrizes)

# 5. Exibindo os resultados
print("=== Classificador de SPAM com Inteligência Artificial ===\n")
for msg, prev in zip(novas_mensagens, previsoes):
    resultado = "🚨 SPAM" if prev == 1 else "✅ NORMAL"
    print(f"[{resultado}] -> {msg}")