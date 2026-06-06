import os
import time
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import google.generativeai as genai

print("🚀 INICIANDO PIPELINE DE MACHINE LEARNING NÃO-SUPERVISIONADO 🚀\n")

# 1. Configuração e Leitura de Dados
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

print("1. Carregando o dataset sintético...")
df = pd.read_csv("reclamacoes_sinteticas.csv")
textos = df['texto_reclamacao'].tolist()

# 2. Geração de Embeddings (Vetorização)
print("2. Carregando motor de Embeddings (all-MiniLM-L6-v2) e vetorizando textos...")
tempo_inicio = time.time()

modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
matriz_embeddings = modelo_embeddings.encode(textos, show_progress_bar=True)

tempo_fim = time.time()
print(f"✅ Embeddings gerados em {tempo_fim - tempo_inicio:.2f} segundos!\n")

# --- AJUSTE DE ENGENHARIA DE ML: REDUÇÃO DE DIMENSIONALIDADE (PCA) ---
print("🔄 Aplicando PCA para reduzir a dimensionalidade de 384 para 10 componentes...")
pca = PCA(n_components=10, random_state=42)
matriz_reduzida = pca.fit_transform(matriz_embeddings)
print("✅ Dimensionalidade reduzida com sucesso.\n")

# 3. Descobrindo o número ideal de Clusters (K)
print("3. Calculando Inércia (Elbow) e Silhueta para k=4 até 12 no espaço reduzido...")
K_range = range(4, 13)
inercia = []
silhuetas = []

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(matriz_reduzida)
    
    inercia.append(kmeans_temp.inertia_)
    silhuetas.append(silhouette_score(matriz_reduzida, labels_temp))

# Plotando e salvando o novo gráfico do Elbow Method (Exigência do PDF)
plt.figure(figsize=(10, 5))
plt.plot(K_range, inercia, marker='o', linestyle='--', color='b')
plt.title('Método do Cotovelo (PCA + Elbow Method)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.grid(True)
plt.savefig('grafico_elbow_method.png')
print("✅ Novo gráfico 'grafico_elbow_method.png' salvo com sucesso.")

# Escolhendo o melhor K automaticamente dentro do intervalo realista baseado na maior silhueta
melhor_k = K_range[np.argmax(silhuetas)]
melhor_score = max(silhuetas)
print(f"🎯 Com o PCA, o modelo escolheu matematicamente k={melhor_k} clusters! (Silhouette Score: {melhor_score:.4f})\n")

# 4. Aplicando o Cluster Definitivo
print(f"4. Agrupando os dados definitivamente em {melhor_k} clusters...")
kmeans_final = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(matriz_reduzida)

# 5. Interpretação com LLM (Dando significado aos números)
print("5. Usando LLM para interpretar os clusters descobertos...\n")
interpretacoes_dos_clusters = {}

for cluster_id in range(melhor_k):
    reclamacoes_cluster = df[df['cluster'] == cluster_id]
    textos_amostra = reclamacoes_cluster['texto_reclamacao'].sample(n=min(5, len(reclamacoes_cluster)), random_state=42).tolist()
    
    prompt = f"""
    Analise estas {len(textos_amostra)} reclamações de clientes de banco que um algoritmo de Machine Learning agrupou por semelhança semântica.
    Textos:
    {json.dumps(textos_amostra, ensure_ascii=False, indent=2)}
    
    Sua tarefa:
    1. Crie um "nome" de negócio bem específico e curto para este cluster (ex: "Fraudes de Pix por Engenharia Social", "Insatisfação com Atendimento Telefônico").
    2. Liste os 3 temas principais abordados.
    3. Qual é o nível de risco predominante para o banco? (Baixo, Médio, Alto).
    
    Retorne APENAS um objeto JSON válido exatamente com esta estrutura:
    {{
      "nome_cluster": "nome curto e direto aqui",
      "temas": ["tema1", "tema2", "tema3"],
      "risco": "Baixo/Médio/Alto"
    }}
    """
    
    try:
        resposta = model.generate_content(prompt)
        texto_limpo = resposta.text.replace("```json", "").replace("```", "").strip()
        interpretacao = json.loads(texto_limpo)
        interpretacoes_dos_clusters[cluster_id] = interpretacao
        
        print(f"Cluster {cluster_id}: {interpretacao['nome_cluster']} (Risco: {interpretacao['risco']})")
        time.sleep(2) # Evita atingir limite da API
    except Exception as e:
        print(f"Erro ao interpretar Cluster {cluster_id}: {e}")
        interpretacoes_dos_clusters[cluster_id] = {"nome_cluster": f"Grupo {cluster_id}", "temas": [], "risco": "Desconhecido"}

# 6. Finalização, Salvamento de Dados e Exportação dos Modelos (Importante para o Agente!)
df['nome_do_cluster_ia'] = df['cluster'].map(lambda x: interpretacoes_dos_clusters[x]['nome_cluster'])
df.to_csv("reclamacoes_com_clusters.csv", index=False, encoding="utf-8")

print("\n💾 Salvando artefatos do modelo para integração com o Agente...")
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans_final, f)

with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

with open("interpretacoes_clusters.json", "w", encoding="utf-8") as f:
    json.dump(interpretacoes_dos_clusters, f, ensure_ascii=False, indent=2)

print("✅ SUCESSO TOTAL! Artefatos ('kmeans_model.pkl', 'pca_model.pkl', 'interpretacoes_clusters.json') salvos.")
print("Pronto para integração com o Agente.")