# aprendizado: Gradient Descent (Descida do Gradiente)

# 1. Definiu erro inicial. 
# (ustudo/suposição: a IA começou chutando um valor ruim, muito alto
peso_atual = 5.0  
taxa_aprendizado = 0.1  #tamanho do passo que a IA dá a cada tentativa (learning Rate)
epocas = 10  # quantidade de vezes que a IA vai tentar aprender com o erro

print(f"Início: Peso inicial = {peso_atual}, Erro inicial = {peso_atual**2:.4f}\n")

# 2. loop do Treinamento
for epoca in range(epocas):
    # Calcula a derivada (a inclinação) da função y = x^2, que é 2x
    gradiente = 2 * peso_atual
    
    # A IA atualiza o peso caminhando na direção CONTRÁRIA ao gradiente (por isso tem o sinal de menos "-")
    peso_atual = peso_atual - (taxa_aprendizado * gradiente)
    
    # Calcula o novo erro
    novo_erro = peso_atual ** 2
    
    print(f"Época {epoca+1}: Gradiente = {gradiente:.2f} | Novo Peso = {peso_atual:.4f} | Novo Erro = {novo_erro:.4f}")

print("\nConclusão: A IA desceu a montanha e reduziu o erro drasticamente!")