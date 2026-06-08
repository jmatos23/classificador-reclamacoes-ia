# FinGuard: Classificador Híbrido de Reclamações com IA

O **FinGuard** é um sistema inteligente baseado em uma arquitetura híbrida que combina o poder analítico de **Machine Learning Não-Supervisionado** (rodando localmente com custo zero) com a capacidade cognitiva e contextual de **Grandes Modelos de Linguagem (LLMs)** via API do Gemini. 

O objetivo principal é identificar padrões ocultos e agrupamentos temáticos em reclamações de clientes bancários, categorizando-as automaticamente e gerando relatórios de risco estruturados em tempo real.

---

## 1. Estrutura do Repositório (Tree View)

Esta é a organização dos arquivos e artefatos gerados ao longo da execução do projeto:

    classificador-reclamacoes-ia/
    ├── .env                  --> Suas chaves protegidas (GEMINI_API_KEY)
    ├── .gitignore            --> Evita subir o .env, venv/ e rascunhos locais
    ├── anotações.txt         --> Seus insights e logs de execução do modelo
    ├── desafio-externo.pdf   --> O escopo original do desafio FinGuard
    │
    ├── generate_dataset.py   --> Geração paralela (15 workers) de 500 reclamações com Gemini 2.5
    ├── reclamacoes_sinteticas.csv --> O dataset bruto gerado pela IA
    │
    ├── pipeline_ml.py        --> Vetorização local + PCA + KMeans + Rotulação via LLM
    ├── grafico_elbow_method.png   --> Gráfico gerado para achar o "K=8" ideal
    ├── reclamacoes_com_clusters.csv --> Base final unindo reclamações e agrupamentos do ML
    │
    ├── kmeans_model.pkl      --> Modelo KMeans treinado e serializado (salvo via pickle)
    ├── pca_model.pkl         --> Redutor de dimensionalidade serializado (salvo via pickle)
    ├── interpretacoes_clusters.json --> Dicionário de conhecimento dos 8 clusters (Nome, Temas, Risco)
    │
    └── agente_analise.py     --> O cérebro final: Agente Híbrido que une ML + Análise Cognitiva

## 2. Diagrama de Fluxo e Arquitetura do Sistema (ASCII Diagram)

O sistema opera em três fases distintas que se conectam por meio de arquivos e artefatos binários (.pkl e .json):

    ========================================================================================
    FASE 1: FÁBRICA DE DADOS (generate_dataset.py)
    ========================================================================================
    
     [ Prompt de Roteiro ] ──> [ ThreadPoolExecutor (15 Workers) ] ──> [ API Gemini 2.5 Flash ]
                                                                                             │
                                                                                             ▼
                                                                    [ reclamacoes_sinteticas.csv ]
    
    ========================================================================================
    FASE 2: PIPELINE DE ML NÃO-SUPERVISIONADO (pipeline_ml.py)
    ========================================================================================
    
      [ reclamacoes_sinteticas.csv ]
                   │
                   ▼
      [ SentenceTransformer ] ───────────> Gera Embeddings Locais (all-MiniLM-L6-v2) de 384 dim.
                   │
                   ▼
      [ PCA (scikit-learn) ] ────────────> Reduz matematicamente para 10 componentes
                   │
                   ▼
      [ KMeans Clustering ] ─────────────> Agrupa os dados em K=8 clusters (Método Elbow/Silhueta)
                   │
                   ▼
      [ API Gemini 2.5 Flash ] ──────────> Lê amostras de cada grupo e cospe JSON estruturado
                   │
                   ▼
      [ SALVAMENTO DE ARTEFATOS ] ───────> Cria: [kmeans_model.pkl], [pca_model.pkl], [interpretacoes_clusters.json]
    
    ========================================================================================
    FASE 3: INFERÊNCIA DO AGENTE HÍBRIDO "FINGUARD" (agente_analise.py)
    ========================================================================================
    
      [ Nova Reclamação Inédita (Texto) ]
                   │
                   ▼
      [ Processamento Numérico Local ]
       ├── 1. SentenceTransformer (Gera vetor)
       ├── 2. Carrega pca_model.pkl (Reduz dim)
       └── 3. Carrega kmeans_model.pkl (Prediz o ID do Cluster)
                   │
                   ▼
      [ Busca de Contexto no JSON ] ─────> Puxa o Nome do Cluster e Risco de [interpretacoes_clusters.json]
                   │
                   ▼
      [ Chamada Cognitiva Final ] ───────> Envia o texto + metadados do ML para o [Gemini 2.5 Flash]
                   │
                   ▼
      [ Relatório Final Híbrido ] ───────> Entrega o JSON mesclando:
                                            ├── Análise Supervisionada (Opinião cognitiva da LLM)
                                            └── Análise Não-Supervisionada (Cluster real do ML matemático)

## 3. Tecnologias Utilizadas e Justificativas

Para garantir uma renderização impecável e elegante na conversão para PDF, as principais ferramentas e fundamentações técnicas do projeto foram catalogadas individualmente em blocos de destaque estruturados:

* **Google Generative AI (Gemini 2.5 Flash)**
    * **Aplicação:** Utilizado com `temperature=0.9` na geração em lote para maximizar a diversidade de personas, e no modo estruturado para rotulação analítica dos clusters.
    * **Justificativa:** Excelente custo-benefício, velocidade de resposta rápida e suporte nativo a retornos estruturados em formato JSON.

* **Sentence Transformers (all-MiniLM-L6-v2)**
    * **Aplicação:** Mapeamento de texto bruto para vetores densos de 384 dimensões.
    * **Justificativa:** Execução 100% local com consumo mínimo de memória RAM, eliminando qualquer custo ou dependência de APIs externas de embeddings.

* **Scikit-Learn (PCA & KMeans Clustering)**
    * **Aplicação:** Redução dimensional matemática (PCA para 10 componentes) seguida pelo agrupamento não-supervisionado das demandas de clientes bancários (K=8).
    * **Justificativa:** Framework robusto padrão de mercado que garante a reprodutibilidade matemática dos agrupamentos por meio de sementes randômicas controladas (SEED = 42).

* **Concurrent Futures (ThreadPoolExecutor)**
    * **Aplicação:** Gerenciamento paralelo de 15 workers simultâneos para requisições de rede.
    * **Justificativa:** Otimização massiva do tempo de execução da fábrica de dados sintéticos, mitigando gargalos de latência de I/O.

## 4. Como Rodar o Projeto do Zero (Guia Passo a Passo)

Siga rigorosamente as etapas visuais abaixo para preparar o seu ambiente de desenvolvimento isolado dentro do editor de código:

**Etapa 1: Preparação de Ambiente e Clonagem**
Abra o seu terminal de comando e inicialize o download seguro do repositório, seguido da abertura automática no editor de texto:

    git clone https://github.com/jmatos23/classificador-reclamacoes-ia.git
    cd classificador-reclamacoes-ia
    code .


**Etapa 2: Instalação da Bolha Isolada (venv)**
Crie e ative um ambiente Python virtual limpo para evitar conflitos de versões globais em sua máquina. Escolha o comando adequado:

*Sistemas Windows (PowerShell/CMD):*

    python -m venv venv
    .\venv\Scripts\activate


*Sistemas Linux / macOS:*

    python3 -m venv venv
    source venv/bin/activate

**Indicador de Sucesso:** Você confirmará a ativação bem-sucedida assim que o prefixo identificador `(venv)` aparecer destacado no canto esquerdo da linha do terminal do VS Code.

**Etapa 3: Carga de Dependências e Bibliotecas**
Com a sua bolha de ambiente ativada, faça a instalação em lote de todos os pacotes matemáticos e utilitários rodando:

    pip install pandas numpy scikit-learn sentence-transformers python-dotenv google-generativeai matplotlib


**Etapa 4: Configuração de Credenciais Seguras**
Crie um arquivo na raiz do seu projeto chamado exatamente `.env`. Adicione a sua credencial privada de acesso à API conforme o modelo abaixo:

    GEMINI_API_KEY=SUA_CHAVE_REAL_DO_GEMINI_AQUI

**Aviso de Segurança Crítico:** Nunca remova o arquivo `.env` do padrão de exclusão do `.gitignore`. Expor chaves privadas de API em repositórios públicos resulta no bloqueio imediato das credenciais por parte do provedor de nuvem.

## 5. Ordem Cronológica de Execução da Pipeline

Para evitar falhas de leitura ou arquivos ausentes, os scripts devem ser executados de forma estritamente sequencial. Cada programa gera a massa de dados que servirá como matéria-prima para a etapa seguinte:

### Tabela de Processamento Sequencial

| Passo | Script de Comando | Atividade Operacional | Artefato Gerado na Raiz |
| :---: | :--- | :--- | :--- |
| **01** | `python generate_dataset.py` | Dispara chamadas de rede em paralelo via *threads* para criar a base inicial fictícia. | `reclamacoes_sinteticas.csv` *(500 registros estruturados)* |
| **02** | `python pipeline_ml.py` | Roda a vetorização sem custo, reduz o espaço vetorial e salva os pesos matemáticos. | `kmeans_model.pkl`<br>`pca_model.pkl`<br>`interpretacoes_clusters.json`<br>`grafico_elbow_method.png` |
| **03** | `python agente_analise.py` | Executa o teste ponta a ponta do agente, aplicando a classificação híbrida (ML + LLM). | **Relatório Consolidado FinGuard** exibido em formato JSON no terminal. |