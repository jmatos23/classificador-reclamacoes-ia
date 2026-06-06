import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Configurações de Segurança e Conexões
load_dotenv()
CHAVE_API = os.getenv("GEMINI_API_KEY")

if not CHAVE_API or CHAVE_API == "cole_sua_chave_aqui_sem_aspas":
    raise ValueError("❌ Erro: Configure sua GEMINI_API_KEY no arquivo .env antes de rodar o Agente!")

genai.configure(api_key=CHAVE_API)
# Usando o Gemini 2.5 Flash para análise estruturada rápida
model = genai.GenerativeModel('gemini-2.5-flash')

# 2. Carregar os Artefatos Matemáticos do Pipeline de ML
print("🛠️ Carregando modelos de Machine Learning...")
try:
    with open("kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    with open("pca_model.pkl", "rb") as f:
        pca_model = pickle.load(f)
    with open("interpretacoes_clusters.json", "r", encoding="utf-8") as f:
        interpretacoes_clusters = json.load(f)
    print("✅ Modelos carregados com sucesso!")
except FileNotFoundError:
    print("❌ Erro: Arquivos do modelo de ML não encontrados!")
    print("Por favor, execute 'python pipeline_ml.py' primeiro para treinar e salvar os modelos.")
    exit(1)

# Carregar o vetorizador (mesmo modelo do pipeline de ML para consistência matemática)
print("🧬 Inicializando tradutor de Embeddings (all-MiniLM-L6-v2)...")
modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Tudo pronto!")


def analisar_reclamacao_completa(texto_reclamacao):
    """
    Recebe um texto livre de reclamação e realiza a análise híbrida:
    1. Classificação Não-Supervisionada (ML local que calcula o cluster correto)
    2. Análise Supervisionada Cognitiva (LLM que extrai metadados em JSON, incluindo o prejuízo)
    """
    
    # === ETAPA 1: MACHINE LEARNING NÃO-SUPERVISIONADO (Local) ===
    # A. Converter texto em vetor (Embedding)
    vetor = modelo_embeddings.encode([texto_reclamacao])
    
    # B. Reduzir dimensionalidade com o PCA carregado
    vetor_reduzido = pca_model.transform(vetor)
    
    # C. Prever o Cluster usando o K-Means carregado
    cluster_previsto = int(kmeans_model.predict(vetor_reduzido)[0])
    
    # D. Buscar a interpretação semântica daquele cluster gerada no pipeline
    info_cluster = interpretacoes_clusters.get(str(cluster_previsto), {
        "nome_cluster": f"Grupo {cluster_previsto}",
        "risco": "Desconhecido"
    })
    
    # === ETAPA 2: AGENTE COGNITIVO SUPERVISIONADO (Gemini API) ===
    prompt_agente = f"""
    Você é um agente especialista em análise de reclamações de clientes bancários.
    Analise o texto abaixo e extraia as informações estruturadas de negócio.
    
    Texto da reclamação:
    "{texto_reclamacao}"
    
    Sua tarefa:
    1. Identifique a Categoria principal (ex: Cobrança Indevida, Atendimento, Fraude/Segurança, Cancelamento, Produto/Serviço, Outros).
    2. Identifique o Produto (ex: Cartão de Crédito, Conta Corrente, Empréstimo ou Vazio se geral).
    3. Determine o Sentimento do cliente (ex: Irritado, Frustrado, Neutro, Sarcástico, Desesperado).
    4. Avalie a Urgência da tratativa (ex: Baixa, Média, Alta).
    5. Crie um Resumo de 1 frase explicando a dor do cliente.
    6. Identifique e extraia o valor numérico do prejuízo financeiro mencionado (ex: valor de Pix fraudulento, compra não reconhecida, cobrança indevida).
       - Retorne APENAS o número puro (ex: 3000 ou 3000.00). Sem símbolos de moeda (R$), sem pontos de milhar, apenas número ou número decimal.
       - Se NÃO houver prejuízo financeiro mencionado no texto, retorne null.
    
    Retorne estritamente um JSON válido seguindo esse formato exato (sem formatação markdown, sem ```json):
    {{
      "categoria": "categoria detectada",
      "produto": "produto detectado",
      "sentimento": "sentimento detectado",
      "urgencia": "urgencia detectada",
      "resumo": "resumo sucinto aqui",
      "prejuizo": 3000.0
    }}
    """
    
    try:
        # Chamada estruturada ao Gemini
        resposta = model.generate_content(
            prompt_agente,
            generation_config={"response_mime_type": "application/json"}
        )
        analise_cognitiva = json.loads(resposta.text.strip())
    except Exception as e:
        print(f"Erro ao processar análise cognitiva da IA: {e}")
        analise_cognitiva = {
            "categoria": "Indefinida",
            "produto": "Indefinido",
            "sentimento": "Neutro",
            "urgencia": "Média",
            "resumo": "Erro ao processar texto com o agente de IA.",
            "prejuizo": None
        }
        
    # === ETAPA 3: INTEGRAÇÃO DOS DOIS MUNDOS ===
    # Unimos a análise do agente com os dados do cluster do ML
    relatorio_final = {
        "analise_supervisionada_agente": analise_cognitiva,
        "analise_nao_supervisionada_ml": {
            "cluster_id": cluster_previsto,
            "nome_cluster": info_cluster["nome_cluster"],
            "risco_predominante_cluster": info_cluster["risco"]
        }
    }
    
    return relatorio_final


# =====================================================================
# ÁREA DE TESTE (Rode este arquivo diretamente para ver o agente funcionando!)
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 TESTANDO O AGENTE FINQUARD INTEGRADO 🤖")
    print("="*50)
    
    # Exemplo de reclamação totalmente inédita de Fraude de Pix
    reclamacao_teste = (
        "Acabei de cair em um golpe horroroso! Recebi uma mensagem dizendo que era do gerente, "
        "cliquei no link e fizeram um Pix de R$ 3.000 da minha conta corrente sem minha permissão. "
        "Quero meu dinheiro de volta urgente ou vou acionar o Banco Central e processar vocês!"
    )
    
    print(f"\nTexto de Entrada:\n'{reclamacao_teste}'\n")
    print("Processando análise integrada...")
    
    resultado = analisar_reclamacao_completa(reclamacao_teste)
    
    # Imprime de forma bonita na tela
    print("\n📊 RESULTADO INTEGRADO (JSON):")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))