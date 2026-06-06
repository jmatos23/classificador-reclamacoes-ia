import os
import random
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Configurações Iniciais e Segurança
load_dotenv()
SEED = 42
random.seed(SEED)

CHAVE_API = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=CHAVE_API)

# Configurando o modelo com Temperature alta (0.9) para forçar criatividade e evitar textos iguais
configuracao_geracao = genai.types.GenerationConfig(temperature=0.9)
model = genai.GenerativeModel('gemini-2.5-flash', generation_config=configuracao_geracao)

print("1. Iniciando a configuração do roteiro...")

TOTAL_RECLAMACOES = 500  
CATEGORIAS = ["Cobrança Indevida", "Atendimento", "Fraude/Segurança", "Produto/Serviço", "Cancelamento", "Outros"]
PESOS = [0.30, 0.20, 0.15, 0.20, 0.10, 0.05]
roteiro_de_geracao = []

for i in range(TOTAL_RECLAMACOES):
    roteiro_de_geracao.append({
        "id_temporario": i + 1,
        "categoria": random.choices(CATEGORIAS, weights=PESOS, k=1)[0],
        "risco_alto": False,
        "dados_sensiveis": False
    })

for idx in random.sample(range(TOTAL_RECLAMACOES), 50): roteiro_de_geracao[idx]["risco_alto"] = True
for idx in random.sample(range(TOTAL_RECLAMACOES), 20): roteiro_de_geracao[idx]["dados_sensiveis"] = True

print("2. Roteiro criado! Iniciando a geração MULTI-PERSONA de dados...")
data_base = datetime(2026, 1, 1)

def gerar_uma_reclamacao(item):
    dias_aleatorios = random.randint(0, 365)
    data_reclamacao = (data_base + timedelta(days=dias_aleatorios)).strftime("%Y-%m-%d")
    id_reclamacao = f"REC-2026-{str(item['id_temporario']).zfill(5)}"
    
    canais = ["SAC", "Ouvidoria", "Banco Central", "Redes Sociais"]
    status_lista = ["Aberta", "Em análise", "Resolvida"]
    produtos = ["Cartão de Crédito", "Conta Corrente", "Empréstimo", ""] 
    
    canal_escolhido = random.choice(canais)
    status_escolhido = random.choice(status_lista)
    produto_escolhido = random.choices(produtos, weights=[0.3, 0.3, 0.3, 0.1], k=1)[0] 

    # --- A NOVA INJEÇÃO DE CAOS PARA EVITAR TEXTOS IGUAIS ---
    personas = [
        "um idoso confuso com tecnologia",
        "um advogado formal e ameaçador",
        "um jovem apressado digitando do celular",
        "um trabalhador simples e indignado",
        "um cliente VIP arrogante",
        "uma pessoa que escreve com erros de português e gírias"
    ]
    tons = [
        "muito irritado e impaciente",
        "extremamente formal e frio",
        "desesperado e pedindo ajuda",
        "irônico e sarcástico",
        "direto ao ponto, sem enrolação"
    ]
    tamanhos = [
        "muito curto (1 ou 2 frases no máximo)",
        "médio (cerca de 3 a 4 frases)",
        "longo e detalhista (vários parágrafos contando a história toda)"
    ]

    persona_escolhida = random.choice(personas)
    tom_escolhido = random.choice(tons)
    tamanho_escolhido = random.choice(tamanhos)

    prompt = f"""
    Aja como {persona_escolhida}. 
    Você está escrevendo uma reclamação para o canal: {canal_escolhido}.
    
    Contexto:
    - Assunto principal: {item['categoria']}
    - Produto envolvido: {produto_escolhido if produto_escolhido else 'Nenhum, reclamação sobre a instituição em geral'}
    - Risco Alto: {item['risco_alto']} (Se True, obrigatoriamente ameace processo judicial, Procon ou Banco Central).
    - Dados Sensíveis: {item['dados_sensiveis']} (Se True, invente e inclua um CPF formato XXX.XXX.XXX-XX ou Conta bancária no texto).
    
    Diretrizes de Estilo:
    - Tom da mensagem: {tom_escolhido}
    - Tamanho: {tamanho_escolhido}
    
    REGRAS ABSOLUTAS:
    1. NUNCA use a palavra "Prezados" ou comece como uma carta formal genérica (a menos que a persona seja o advogado).
    2. NUNCA use placeholders como "[Seu Nome]", "[Nome da Empresa]". Invente um nome falso tipo João, Maria, etc., ou simplesmente não assine.
    3. Pareça uma pessoa real da internet escrevendo.
    
    RETORNE APENAS O TEXTO DA RECLAMAÇÃO. Sem aspas, sem markdown.
    """
    
    texto_gerado = "Erro na geração"
    max_tentativas = 3
    for tentativa in range(max_tentativas):
        try:
            resposta = model.generate_content(prompt)
            texto_gerado = resposta.text.strip()
            break 
        except Exception as e:
            time.sleep(1) 
            if tentativa == max_tentativas - 1:
                texto_gerado = f"Erro na API. Categoria: {item['categoria']}"

    print(f"✅ Item {item['id_temporario']} gerado! (Persona: {persona_escolhida[:15]}...)")
    
    return {
        "id": id_reclamacao,
        "data_reclamacao": data_reclamacao,
        "canal": canal_escolhido,
        "texto_reclamacao": texto_gerado,
        "produto": produto_escolhido,
        "status": status_escolhido,
        "categoria_oculta_gabarito": item['categoria'] 
    }

dados_finais = []

with ThreadPoolExecutor(max_workers=15) as executor:
    futuros = [executor.submit(gerar_uma_reclamacao, item) for item in roteiro_de_geracao]
    for futuro in as_completed(futuros):
        dados_finais.append(futuro.result())

dados_finais = sorted(dados_finais, key=lambda x: x['id'])

print("3. Salvando o arquivo CSV...")
df = pd.DataFrame(dados_finais)
df.to_csv("reclamacoes_sinteticas.csv", index=False, encoding="utf-8")

print(f"✅ SUCESSO! Dataset gerado com alta variância e sem textos robóticos repetidos.")