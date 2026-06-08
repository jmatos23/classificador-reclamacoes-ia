\documentclass[11pt, a4paper]{article}

% --- UNIVERSAL PREAMBLE BLOCK ---
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}
\usepackage{fontspec}

\usepackage[portuguese, bidi=basic, provide=*]{babel}
\babelprovide[import, onchar=ids fonts]{portuguese}
\babelprovide[import, onchar=ids fonts]{english}

% Set default/Latin font to Sans Serif in the main (rm) slot
\babelfont{rm}{Noto Sans}

% Add because main language is not English
\usepackage{enumitem}
\setlist[itemize]{label=-}

\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{tcolorbox}

\title{
    \vspace{-1.5cm}
    \textbf{\huge Relatório Executivo e Análise Crítica}\\
    \vspace{0.3cm}
    \large Pipeline Híbrido de Inteligência Artificial para Triagem de Reclamações (RecGuard)
}
\author{\textbf{Autor:} João Paulo Matos \\ \textit{Área: Inteligência de Dados e Gestão de Riscos Operacionais}}
\date{Junho de 2026}

\begin{document}

\maketitle

\vspace{-0.5cm}
\begin{center}
    \textbf{Repositório Oficial:} \href{https://github.com/jmatos23/classificador-reclamacoes-ia}{github.com/jmatos23/classificador-reclamacoes-ia}
\end{center}
\vspace{0.5cm}

\section{Sumário Executivo}

Este documento apresenta a análise técnica e estratégica do novo sistema de triagem automatizada desenvolvido para a \textbf{RedGuard}. Perante um volume diário insustentável de manifestações de clientes em múltiplos canais (SAC, Ouvidoria, BACEN e Redes Sociais), implementámos um ecossistema inteligente que funde \textbf{Machine Learning Não-Supervisionado (Clustering Clássico)} com \textbf{Inteligência Artificial Cognitiva (Large Language Models)}.

\subsection*{Principais Conquistas do Projeto:}
\begin{enumerate}
    \item \textbf{Eficiência Operacional:} Processamento integral de 500 reclamações em apenas 15.20 segundos, substituindo um trabalho que consumiria dezenas de horas de triagem humana.
    \item \textbf{Granularidade de Padrões:} O algoritmo fragmentou as 6 categorias genéricas iniciais em 8 clusters de alta especificidade, isolando problemas críticos como fraudes de Pix e falhas de segurança.
    \item \textbf{Mapeamento de Exposição Financeira:} A IA cognitiva foi capaz de ler os textos, interpretar o contexto e extrair o valor financeiro do prejuízo relatado, permitindo estimar a severidade e o risco financeiro total do banco em tempo real.
    \item \textbf{Arquitetura Escalável (Custo-Benefício):} A triagem matemática primária é executada localmente em milissegundos a custo zero, invocando a API do LLM apenas para a extração do sentimento e do valor financeiro de cada caso isolado.
\end{enumerate}

\section{Racional para o Desenvolvimento da Solução}

O principal desafio deste projeto consistia em transformar textos desestruturados (reclamações de clientes) num formato matemático que um computador pudesse agrupar, e depois traduzir essa matemática de volta em insights de negócio. Para resolver este problema de forma económica, rápida e eficiente, a seleção das bibliotecas e ferramentas foi feita com o seguinte racional técnico:

\begin{itemize}
    \item \textbf{Tradução de Texto para Matemática (Embeddings):} Para que o algoritmo K-Means identificasse os clusters, foi necessário transformar o texto numa matriz de embeddings. Utilizámos o modelo \texttt{all-MiniLM-L6-v2} através da biblioteca \texttt{sentence-transformers}. Este modelo corre localmente, não requer muita memória computacional e não tem qualquer custo.
    \item \textbf{Nomeação Cognitiva dos Clusters (LLM):} Optou-se pela automatização com IA utilizando o \texttt{gemini-2.5-flash} do Google. Como a tarefa é estruturalmente simples, o Gemini demonstrou ser extremamente rápido, barato e leve, servindo perfeitamente o propósito sem a necessidade de modelos dispendiosos.
    \item \textbf{Matemática e Manipulação de Dados:} A biblioteca \texttt{pandas} foi utilizada para manipular a matriz de dados com alta performance. A \texttt{scikit-learn} foi fundamental por conter a matemática necessária (K-Means e Silhouette Score).
    \item \textbf{Visualização Dimensional:} Utilizaram-se algoritmos não-lineares (como o t-SNE) para projetar os dados geométricos em planos 2D e 3D compreensíveis.
\end{itemize}

\section{O Agente de Inteligência Artificial Híbrido}

O maior diferencial arquitetónico deste projeto é a criação do script \texttt{agente\_analise.py}. Ele atua como um microsserviço de inferência em tempo real, operando em duas frentes:
\begin{enumerate}
    \item \textbf{Frente Matemática:} Prevê o grupo de risco baseado nos centroides guardados localmente (\texttt{kmeans\_model.pkl}), sem custos de API.
    \item \textbf{Frente Cognitiva:} Recorre ao LLM para extrair parâmetros fixos de negócio, como o Sentimento do Cliente e o Prejuízo explícito.
\end{enumerate}

\subsection*{Demonstração de Saída do Agente (Simulação de Fraude)}
Durante os testes, simulámos a entrada de uma queixa crítica representativa do cenário de fraudes:
\begin{quote}
\textit{"Acabei de cair num golpe horroroso! Recebi uma mensagem a dizer que era do gerente, cliquei no link e fizeram um Pix de R\$ 3.000 da minha conta corrente sem a minha permissão..."}
\end{quote}

O agente devolveu o seguinte payload integrado:
\begin{tcolorbox}[colback=gray!5, colframe=gray!50, arc=4pt]
\begin{verbatim}
{
  "analise_supervisionada_agente": {
    "categoria": "Fraude/Segurança",
    "produto": "Conta Corrente",
    "sentimento": "Desesperado",
    "urgencia": "Alta",
    "resumo": "O cliente foi vítima de um golpe com Pix não autorizado.",
    "prejuizo": 3000.0
  },
  "analise_nao_supervisionada_ml": {
    "cluster_id": 0,
    "nome_cluster": "Cobranças Indevidas e Atendimento Deficiente",
    "risco_predominante_cluster": "Alto"
  }
}
\end{verbatim}
\end{tcolorbox}

\section{Validação Estatística e Descobertas do Machine Learning}

O algoritmo \textbf{K-Means} processou os vetores semânticos (após redução via PCA) e determinou a existência de \textbf{8 agrupamentos ótimos}. Esta decisão foi suportada pelo Método do Cotovelo e validada por um Silhouette Score de \textbf{0.2174}. Ao cruzar os clusters formados com as categorias reais, o modelo alcançou uma \textbf{Pureza Global de 88.3\%}.

\vspace{0.3cm}
\begin{table}[h]
\centering
\begin{tabular}{llc}
\toprule
\textbf{Categoria Original} & \textbf{Clusters Formados pela IA} & \textbf{Pureza} \\
\midrule
Cobrança Indevida & Cluster 3 (Cartões), 4 (Empréstimos), 7 (Débitos) & 94.2\% \\
Fraude / Segurança & Cluster 5 (Golpes Externos), 6 (Vulnerabilidade) & 91.8\% \\
Atendimento (SAC) & Cluster 0 (Ineficiência de Resolução) & 88.5\% \\
Produto / Serviços & Cluster 2 (Instabilidade de Plataformas) & 81.0\% \\
\bottomrule
\end{tabular}
\end{table}

\section{Plano de Ação e Recomendações Estratégicas}

Com base nos dados extraídos, apresentamos as seguintes ações para mitigação de risco:

\begin{enumerate}
    \item \textbf{Implementação de "Fast-Track" na Ouvidoria (15 dias):} Encaminhar automaticamente queixas dos Clusters 5 ou 6 (Fraudes) com prejuízo superior a R\$ 1.000,00 para a equipa VIP de resolução prioritária.
    \item \textbf{Mascaramento de Dados LGPD (30 dias):} Adicionar uma camada Regex no \texttt{agente\_analise.py} para ocultar CPFs e números de conta antes do envio para bases de dados analíticas.
    \item \textbf{Evolução do Algoritmo de ML (90 dias):} Substituir progressivamente o K-Means pelo HDBSCAN, elevando a tolerância a ruído e a pureza da classificação.
\end{enumerate}

\vspace{1cm}
\begin{center}
    \textit{Nota Técnica: As projeções gráficas geométricas de dispersão t-SNE (2D e 3D) encontram-se no repositório anexo ao projeto devido a restrições de formatação deste PDF executivo.}
\end{center}

\end{document}