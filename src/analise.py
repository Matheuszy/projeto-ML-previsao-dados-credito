import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 


def carregar_dados(caminho_arquivo):
    """
    Carrega um arquivo CSV em um DataFrame Pandas.

    Args:
        caminho_arquivo (str): O caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: O DataFrame carregado. Retorna None em caso de erro.
    """
    print(f"Carregando dados de: {caminho_arquivo}")
    try:
        df = pd.read_csv(caminho_arquivo)
        print("Dados carregados com sucesso!")
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado. Verifique o caminho.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo: {e}")
        return None


def pre_processar_dados(df, colunas_para_codificar, codificadores=None, fit_codificadores=True):
    """
    Aplica Label Encoding em colunas categóricas de um DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame a ser pré-processado.
        colunas_para_codificar (list): Lista de nomes das colunas categóricas.
        codificadores (dict, optional): Um dicionário de codificadores LabelEncoder pré-treinados.
                                        Se None, novos codificadores serão criados.
        fit_codificadores (bool): Se True, os codificadores serão treinados (fit) nos dados.
                                  Deve ser True para dados de treino e False para dados de teste/novos dados.

    Returns:
        tuple: Um tuple contendo o DataFrame processado e o dicionário de codificadores (novos ou passados).
    """
    if df is None:
        return None, codificadores

    if codificadores is None:
        codificadores = {}

    df_processado = df.copy() 

    print("\nIniciando pré-processamento de dados (Label Encoding)...")
    for coluna in colunas_para_codificar:
        if coluna in df_processado.columns:
            if fit_codificadores:
                
                codificador = LabelEncoder()
                df_processado[coluna] = codificador.fit_transform(df_processado[coluna])
                codificadores[coluna] = codificador 
                print(f"Coluna '{coluna}' codificada e codificador treinado.")
            else:
                
                if coluna in codificadores:
                    
                    df_processado[coluna] = codificadores[coluna].transform(df_processado[coluna])
                    print(f"Coluna '{coluna}' codificada usando codificador existente.")
                else:
                    print(f"Aviso: Codificador para '{coluna}' não encontrado para transformação. Coluna não codificada.")
        else:
            print(f"Aviso: Coluna '{coluna}' não encontrada no DataFrame para codificação.")

    return df_processado, codificadores


def treinar_modelos(x_treino, y_treino):
    """
    Treina modelos de Random Forest e K-Nearest Neighbors.

    Args:
        x_treino (pd.DataFrame): Features para treinamento.
        y_treino (pd.Series): Target para treinamento.

    Returns:
        tuple: Uma tupla contendo os modelos treinados (RandomForestClassifier, KNeighborsClassifier).
    """
    print("\nIniciando treinamento dos modelos...")

    modelo_arvore_decisao = RandomForestClassifier(random_state=42) # Adicionei random_state para reprodutibilidade
    modelo_vizinhos_proximos = KNeighborsClassifier()

    modelo_arvore_decisao.fit(x_treino, y_treino)
    print("Modelo Random Forest treinado com sucesso.")

    modelo_vizinhos_proximos.fit(x_treino, y_treino)
    print("Modelo K-Nearest Neighbors treinado com sucesso.")

    return modelo_arvore_decisao, modelo_vizinhos_proximos


def avaliar_modelos(modelos, x_teste, y_teste):
    """
    Avalia a acurácia de uma lista de modelos.

    Args:
        modelos (list): Uma lista de tuplas (nome_modelo, modelo_treinado).
        x_teste (pd.DataFrame): Features para teste.
        y_teste (pd.Series): Target real para teste.
    """
    print("\n--- Avaliação dos Modelos ---")
    resultados = {}
    for nome, modelo in modelos:
        previsoes = modelo.predict(x_teste)
        acuracia = accuracy_score(y_teste, previsoes)
        resultados[nome] = acuracia
        print(f"Acurácia do {nome}: {acuracia:.4f}")
    return resultados


def fazer_previsao(modelo, df_novos_clientes):
    """
    Faz previsões em um novo DataFrame de clientes usando um modelo treinado.

    Args:
        modelo: O modelo de Machine Learning treinado.
        df_novos_clientes (pd.DataFrame): DataFrame com os novos dados de clientes pré-processados.

    Returns:
        numpy.ndarray: Array contendo as previsões do score de crédito.
    """
    if df_novos_clientes is None:
        print("Não é possível fazer previsões. DataFrame de novos clientes é nulo.")
        return None

    print("\nRealizando previsões em novos clientes...")
    previsoes = modelo.predict(df_novos_clientes)
    print("Previsões concluídas!")
    return previsoes


def main():
    """
    Função principal que orquestra toda a pipeline de análise de dados.
    """
    
    caminho_clientes = "../dados/clientes.csv"
    caminho_novos_clientes = "../dados/novos_clientes.csv"
    colunas_categoricas = ['profissao', 'mix_credito', 'comportamento_pagamento']

    
    df_clientes = carregar_dados(caminho_clientes)
    if df_clientes is None:
        return

    
    print("\n--- Dados Originais ('clientes.csv') ---")
    print(df_clientes.head()) 
    print("\nInformações do DataFrame Original:")
    df_clientes.info()

    
    df_clientes_processado, codificadores_treinados = pre_processar_dados(
        df_clientes, colunas_categoricas, fit_codificadores=True
    )
    if df_clientes_processado is None:
        return

    print("\n--- Dados Pré-processados ('clientes.csv') ---")
    print(df_clientes_processado.head()) 

    
    colunas_para_dropar = ['score_credito']
    if 'id_cliente' in df_clientes_processado.columns:
        colunas_para_dropar.append('id_cliente')

    y = df_clientes_processado['score_credito']
    x = df_clientes_processado.drop(columns=colunas_para_dropar)


    print("\nDividindo dados em conjuntos de treino e teste...")
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Tamanho do conjunto de treino: {len(x_treino)} amostras")
    print(f"Tamanho do conjunto de teste: {len(x_teste)} amostras")

    
    modelo_arvore_decisao, modelo_vizinhos_proximos = treinar_modelos(x_treino, y_treino)

    
    modelos_para_avaliar = [
        ("Random Forest", modelo_arvore_decisao),
        ("K-Nearest Neighbors", modelo_vizinhos_proximos)
    ]
    avaliar_modelos(modelos_para_avaliar, x_teste, y_teste)

    
    melhor_modelo_para_previsao = modelo_arvore_decisao
    print("\nSelecionado Random Forest como o modelo para previsão.")

    
    df_novos_clientes = carregar_dados(caminho_novos_clientes)
    if df_novos_clientes is None:
        return

    print("\n--- Novos Dados ('novos_clientes.csv') antes do pré-processamento ---")
    print(df_novos_clientes.head()) 

    
    df_novos_clientes_processado, _ = pre_processar_dados(
        df_novos_clientes, colunas_categoricas, codificadores=codificadores_treinados, fit_codificadores=False
    )
    if df_novos_clientes_processado is None:
        return

   
    if 'id_cliente' in df_novos_clientes_processado.columns:
        df_novos_clientes_processado = df_novos_clientes_processado.drop(columns=['id_cliente'])

    print("\n--- Novos Dados Pré-processados ('novos_clientes.csv') ---")
    print(df_novos_clientes_processado.head()) 

    
    previsao_novos_clientes = fazer_previsao(melhor_modelo_para_previsao, df_novos_clientes_processado)

    print("\n--- Previsões para Novos Clientes ---")
    
    print(previsao_novos_clientes)

    
    df_novos_clientes['score_credito_previsto'] = previsao_novos_clientes
    print("\n--- Novos Clientes com Previsões ---")
    print(df_novos_clientes) 


# --- Execução Principal ---
if __name__ == "__main__":
    main()