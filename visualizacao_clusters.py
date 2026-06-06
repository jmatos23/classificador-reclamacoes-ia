import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
# Importação explícita do motor de projeção 3D do Matplotlib
from mpl_toolkits.mplot3d import Axes3D

print("--- INICIANDO PROTOCOLO DE REPRESENTAÇÃO GRÁFICA (2D E 3D) ---\n")

# 1. Carregar os dados agrupados (clusters)
if not os.path.exists("reclamacoes_com_clusters.csv"):
    print("❌ Erro: O arquivo 'reclamacoes_com_clusters.csv' não foi encontrado!")
    print("Execute o 'pipeline_ml.py' primeiro para gerar os clusters.")
    exit(1)

print("1. Lendo os dados com clusters salvos...")
df = pd.read_csv("reclamacoes_com_clusters.csv")

# 2. Gerar novamente os Embeddings para projeção espacial
# (Precisamos dos vetores originais para calcular as distâncias do t-SNE)
print("2. Gerando embeddings para o mapeamento espacial (all-MiniLM-L6-v2)...")
modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
matriz_embeddings = modelo_embeddings.encode(df['texto_reclamacao'].tolist(), show_progress_bar=True)

# 3. Processamento do t-SNE para 2D e 3D
# Justificação Estatística: Rodamos o t-SNE duas vezes separadamente. 
# Embora pudéssemos apenas pegar 2 eixos do 3D, rodar o t-SNE configurado para n_components=2 
# garante que o gráfico bidimensional tenha a menor perda de divergência de Kullback-Leibler possível para 2D.
print("\n3. Projetando 384 dimensões usando t-SNE para 2D e 3D...")

print("   -> Calculando projeção 2D...")
tsne_2d = TSNE(
    n_components=2, 
    perplexity=30, 
    random_state=42, 
    learning_rate='auto'
)
coordenadas_2d = tsne_2d.fit_transform(matriz_embeddings)
df['tsne_2d_x'] = coordenadas_2d[:, 0]
df['tsne_2d_y'] = coordenadas_2d[:, 1]

print("   -> Calculando projeção 3D...")
tsne_3d = TSNE(
    n_components=3, 
    perplexity=30, 
    random_state=42, 
    learning_rate='auto'
)
coordenadas_3d = tsne_3d.fit_transform(matriz_embeddings)
df['tsne_3d_x'] = coordenadas_3d[:, 0]
df['tsne_3d_y'] = coordenadas_3d[:, 1]
df['tsne_3d_z'] = coordenadas_3d[:, 2]

# Configurações comuns de cores e mapeamento de nomes
unique_clusters = sorted(df['cluster'].unique())
paleta_cores = sns.color_palette("Set2", n_colors=len(unique_clusters))
mapeamento_nomes = df.set_index('cluster')['nome_do_cluster_ia'].to_dict()
df['Nome do Cluster (IA)'] = df['cluster'].map(mapeamento_nomes)

# --- 4. CONSTRUINDO E SALVANDO O GRÁFICO 2D ---
print("\n4. Renderizando e salvando o gráfico de dispersão 2D...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 10))

scatter_2d = sns.scatterplot(
    data=df,
    x='tsne_2d_x',
    y='tsne_2d_y',
    hue='Nome do Cluster (IA)',
    palette=paleta_cores,
    alpha=0.85,
    s=80,
    edgecolor='w',
    linewidth=0.5
)

plt.title(
    'Mapeamento Semântico de Reclamações de Clientes (Projeção t-SNE 2D)', 
    fontsize=16, 
    fontweight='bold', 
    pad=20
)
plt.xlabel('Dimensão Espacial t-SNE X', fontsize=12, labelpad=10)
plt.ylabel('Dimensão Espacial t-SNE Y', fontsize=12, labelpad=10)

plt.legend(
    title='Clusters Identificados pelo K-Means', 
    title_fontsize=12,
    bbox_to_anchor=(1.02, 1), 
    loc='upper left', 
    borderaxespad=0,
    fontsize=10,
    frameon=True,
    shadow=True
)

plt.tight_layout()
caminho_salvamento_2d = "visualizacao_2d_clusters.png"
plt.savefig(caminho_salvamento_2d, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Arquivo 2D salvo com sucesso em: '{caminho_salvamento_2d}'")


# --- 5. CONSTRUINDO E SALVANDO O GRÁFICO 3D ---
print("\n5. Renderizando e salvando o gráfico de dispersão 3D...")
sns.set_theme(style="white")
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# Desenhamos cada cluster individualmente para manter a legenda idêntica e organizada
for i, cluster_id in enumerate(unique_clusters):
    sub_df = df[df['cluster'] == cluster_id]
    nome_ia = mapeamento_nomes.get(cluster_id, f"Grupo {cluster_id}")
    ax.scatter(
        sub_df['tsne_3d_x'],
        sub_df['tsne_3d_y'],
        sub_df['tsne_3d_z'],
        color=paleta_cores[i],
        label=nome_ia,
        alpha=0.8,
        s=60,
        edgecolor='w',
        linewidth=0.3
    )

ax.set_title(
    'Mapeamento Semântico de Reclamações (Projeção t-SNE 3D)', 
    fontsize=16, 
    fontweight='bold', 
    pad=20
)
ax.set_xlabel('Dimensão Espacial t-SNE X', fontsize=11, labelpad=10)
ax.set_ylabel('Dimensão Espacial t-SNE Y', fontsize=11, labelpad=10)
ax.set_zlabel('Dimensão Espacial t-SNE Z', fontsize=11, labelpad=10)

# Ajustar o ângulo de inclinação e rotação da câmera
ax.view_init(elev=25, azim=45)

ax.legend(
    title='Clusters Identificados pelo K-Means', 
    title_fontsize=12,
    bbox_to_anchor=(1.05, 0.9), 
    loc='upper left', 
    borderaxespad=0,
    fontsize=10,
    frameon=True,
    shadow=True
)

plt.tight_layout()
caminho_salvamento_3d = "visualizacao_3d_clusters.png"
plt.savefig(caminho_salvamento_3d, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Arquivo 3D salvo com sucesso em: '{caminho_salvamento_3d}'")

print("\n🎉 PROCEDIMENTO CONCLUÍDO! Ambas as representações gráficas foram salvas!")