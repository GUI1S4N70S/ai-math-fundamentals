import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 1. GERANDO O DATASET (Simulando 1.000 clientes)
X, y = make_classification(n_samples=1000, n_features=4, n_informative=3, n_redundant=1, random_state=42)
df = pd.DataFrame(X, columns=['Meses_Contrato', 'Valor_Mensal', 'Chamados_Suporte', 'Atrasos_Pagamento'])
df['Churn'] = y # 1 = Cancelou, 0 = Continuou

print("=== Amostra dos Dados dos Clientes ===")
print(df.head(), "\n")

# 2. SEPARANDO TREINO E TESTE (A Prova Surpresa)
# 80% para a IA estudar, 20% para testarmos se ela aprendeu ou apenas decorou
X_treino, X_teste, y_treino, y_teste = train_test_split(df.drop('Churn', axis=1), df['Churn'], test_size=0.2, random_state=42)

# 3. DEFININDO OS 3 ALGORITMOS ARQUITETURAIS
modelos = {
    "Regressão Logística": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# 4. TREINAMENTO E AVALIAÇÃO DE MODELOS
print("=== Resultados da Avaliação de Modelos ===")
for nome, modelo in modelos.items():
    # Treinando a IA
    modelo.fit(X_treino, y_treino)
    
    # Fazendo previsões nos dados que ela nunca viu (X_teste)
    previsoes = modelo.predict(X_teste)
    probabilidades = modelo.predict_proba(X_teste)[:, 1] # Necessário para a métrica AUC
    
    # Calculando as métricas exigidas na Fase 1
    precisao = precision_score(y_teste, previsoes)
    revocacao = recall_score(y_teste, previsoes)
    f1 = f1_score(y_teste, previsoes)
    auc = roc_auc_score(y_teste, probabilidades)
    
    # Exibindo o placar
    print(f"🏆 {nome}:")
    print(f"   - Precision: {precisao:.2f}")
    print(f"   - Recall:    {revocacao:.2f}")
    print(f"   - F1-Score:  {f1:.2f}")
    print(f"   - AUC:       {auc:.2f}\n")