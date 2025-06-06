# 🏦 Análise e Previsão de Score de Crédito Bancário

Este projeto implementa uma pipeline de Machine Learning para prever o score de crédito de clientes bancários, auxiliando na tomada de decisões estratégicas e na mitigação de riscos financeiros.

---

## 📊 Visão Geral do Projeto

A pipeline de Machine Learning desenvolvida aqui automatiza o processo de previsão de score de crédito. Ela carrega dados de clientes, realiza o pré-processamento necessário, treina modelos de classificação e, por fim, utiliza o modelo treinado para prever o score de crédito de novos clientes.

Utilizamos Python com as seguintes bibliotecas:
* **Pandas:** Para manipulação e análise de dados.
* **Scikit-learn:** Para pré-processamento de dados (LabelEncoder), divisão de dados (train_test_split), treinamento de modelos (RandomForestClassifier, KNeighborsClassifier) e avaliação (accuracy_score).

---

## 📁 Estrutura do Projeto

├── .venv/                         # Ambiente virtual Python
├── dados/
│   ├── clientes.csv               # Conjunto de dados histórico de clientes (para treino e teste)
│   └── novos_clientes.csv         # Conjunto de dados de novos clientes (para previsão)
└── src/
└── pipeline_credito.py        # Script principal da pipeline de Machine Learning
├── .gitignore                     # Arquivo para ignorar arquivos e pastas no controle de versão (Git)
├── README.md                      # Este arquivo de documentação
└── requirements.txt               # Dependências do projeto


## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e rodar a pipeline de previsão de crédito:

1.  **Clone o Repositório (se estiver em Git):**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd seu_projeto
    ```
2.  **Crie e Ative o Ambiente Virtual:**
    É altamente recomendável usar um ambiente virtual para gerenciar as dependências do projeto.
    ```bash
    python -m venv .venv
    # No Windows:
    .venv\Scripts\activate
    # No macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Instale as Dependências:**
    Com o ambiente virtual ativado, instale todas as bibliotecas necessárias.
    ```bash
    pip install -r requirements.txt
    ```
    *Certifique-se de que seu `requirements.txt` contenha `pandas` e `scikit-learn`.*

4.  **Prepare os Dados:**
    Garanta que os arquivos `clientes.csv` (com dados históricos e score de crédito) e `novos_clientes.csv` (com dados de clientes para previsão, sem o score de crédito) estejam dentro da pasta `dados/`.

5.  **Execute a Pipeline de Previsão:**
    Para rodar a análise de dados, treinar os modelos e obter as previsões para novos clientes, execute o script principal:
    ```bash
    python src/pipeline_credito.py
    ```
    *As saídas, como informações dos DataFrames e as previsões, serão impressas diretamente no seu terminal.*

---

## ⚙️ Detalhes Técnicos da Pipeline

A pipeline executa as seguintes etapas:

1.  **Carregamento de Dados:** Lê `clientes.csv` e `novos_clientes.csv`.
2.  **Pré-processamento:**
    * Identifica colunas categóricas (`profissao`, `mix_credito`, `comportamento_pagamento`).
    * Aplica **Label Encoding** para converter essas colunas em representações numéricas, garantindo que os mesmos mapeamentos sejam usados consistentemente entre os dados de treino e os novos dados.
    * Remove a coluna `id_cliente`, que não é útil para o treinamento do modelo.
3.  **Divisão de Dados:** Separa o conjunto de dados `clientes.csv` em conjuntos de treino e teste (80/20) para treinar e avaliar os modelos.
4.  **Treinamento de Modelos:**
    * **RandomForestClassifier:** Treina um modelo de Random Forest.
    * **KNeighborsClassifier:** Treina um modelo K-Nearest Neighbors.
5.  **Avaliação:** Calcula e exibe a **acurácia** de cada modelo no conjunto de teste.
6.  **Previsão:** Utiliza o modelo de **Random Forest** (que geralmente oferece bom desempenho em dados tabulares) para prever o `score_credito` para cada cliente no arquivo `novos_clientes.csv`. As previsões são então adicionadas ao DataFrame de novos clientes.

---

## 🤝 Contribuições

Sinta-se à vontade para abrir `issues` ou enviar `pull requests` se tiver sugestões de melhorias ou encontrar algum problema.

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT.