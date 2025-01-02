import glob
import random
import itertools
import numpy as np
import pandas as pd
import statistics as stat
from itertools import *
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class loto:

  def __init__(self, loteria):
    self.jlist = []
    self.ini = 1
    self.qtd_jogos = 10
    self.read_results(loteria+'_*')

    if loteria == 'mega_sena':
      self.fim = 60
      self.num_sort = 6
      
      self.df['acertos'] = 6
      self.qtd_num_int = 4

    elif loteria == 'loto_facil':
      self.fim = 25
      self.num_sort = 15

      self.df['acertos'] = 15
      self.qtd_num_int = 10

    elif loteria == 'quina':
      self.fim = 80
      self.num_sort = 5

      self.df['acertos'] = 5

    else:
      print('Loteria não encontrada')

    self.df = self.df.drop(columns=['Concurso', 'Data'])
  
  def read_results(self, arquivo):
    
    # Localizar o arquivo
    arquivos = glob.glob(f"{arquivo}.xlsx")
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão: {arquivo}")

    # Carregar o primeiro arquivo encontrado
    arquivo_excel = arquivos[0]
    df_raw = pd.read_excel(arquivo_excel)

    # Identificar a linha que contém os nomes das colunas (dados relevantes geralmente começam após uma linha específica)
    for index, row in df_raw.iterrows():
        if "Concurso" in row.values:  # Procura pela palavra-chave no cabeçalho
            linha_cabecalho = index
            break

    # Recarregar os dados a partir da linha correta
    self.df = pd.read_excel(arquivo_excel, skiprows=linha_cabecalho+1)

  def testa_n_pontos_em_comum(self, j1, j2, n):
    setj1 = set(j1)
    setj2 = set(j2)
    if len(setj1.intersection(setj2)) >= n:
      return True
    else:
      return False

  def gera_lista_aleatoria_de_elementos_nao_repetidos(self):
    unique_ints = set()
    while len(unique_ints) < self.num_sort:
      unique_ints.add(random.randint(self.ini, self.fim))
    return list(unique_ints)
  
  def gera_jogos_com_max_n_numeros_em_comum(self, n):
    """
    Gera uma lista de jogos aleatórios, garantindo que cada jogo tenha no máximo `n` números em comum
    com qualquer outro jogo da lista.
    
    Parâmetros:
        n (int): Número máximo de elementos que podem coincidir entre dois jogos.

    Retorna:
        list: Lista de jogos gerados.
    """
    def tem_excesso_de_comuns(jogo, jogos_existentes, max_comuns):
        """
        Verifica se o jogo tem mais de `max_comuns` números em comum com algum dos jogos existentes.
        """
        for jogo_existente in jogos_existentes:
            comuns = len(set(jogo) & set(jogo_existente))
            if comuns > max_comuns:
                return True
        return False

    # Gera o primeiro jogo aleatório
    self.jlist = [self.gera_lista_aleatoria_de_elementos_nao_repetidos()]
    
    # Gerar os próximos 10 jogos
    while len(self.jlist) < self.qtd_jogos:
        jaux = self.gera_lista_aleatoria_de_elementos_nao_repetidos()
        # Verifica se o jogo gerado atende à condição de no máximo `n` números em comum
        if not tem_excesso_de_comuns(jaux, self.jlist, n):
            self.jlist.append(jaux)
    
    return self.jlist

  def aumentar_dados(self):
    """
    Aumenta os dados gerando novos jogos e calculando o maior número de acertos
    em relação aos jogos existentes no DataFrame original.
    """
    n, m = self.df.shape
    # Transformar os jogos existentes em uma lista de conjuntos para interseção rápida
    jogos_existentes = [set(self.df.iloc[i, :-1]) for i in range(n)]
    
    add_dados = []  # Lista para armazenar os novos jogos e seus acertos
    
    for _ in range(n):  # Itera sobre o número de jogos existentes
        novos_jogos = self.gera_jogos_com_max_n_numeros_em_comum(self.qtd_num_int)  # Gera novos jogos
        for jogo in novos_jogos:
            jogo_set = set(jogo)  # Converte o jogo gerado em um conjunto para interseção rápida
            
            # Calcular o maior número de acertos com os jogos existentes
            max_acertos = max(len(jogo_set & jogo_existente) for jogo_existente in jogos_existentes)
            
            # Adiciona o novo jogo e o maior acerto à lista
            add_dados.append(jogo + [max_acertos])
    
    # Cria um DataFrame para os novos jogos
    colunas = [f"bola {i+1}" for i in range(self.num_sort)] + ["acertos"]
    df_novos_jogos = pd.DataFrame(add_dados, columns=colunas)
    
    # Concatena os novos jogos ao DataFrame original
    self.df = pd.concat([self.df, df_novos_jogos], ignore_index=True)

  def treinar_modelo(self):
    
    self.aumentar_dados()

    X = self.df.drop('acertos', axis=1)
    y = self.df['acertos']

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dicionário de modelos
    models = {
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),  # Ajustado para convergência
        'SVC': SVC(probability=True, random_state=42),  # Support Vector Classifier com probabilidade
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),  # K-Nearest Neighbors com 5 vizinhos
        'GaussianNB': GaussianNB(),  # Naive Bayes
        'AdaBoostClassifier': AdaBoostClassifier(random_state=42)  # AdaBoost com estimadores fracos
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy}")

    # Find the best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name}")

    # Treinar o melhor modelo (definido previamente)
    best_model.fit(X_train, y_train)

    # Gerar jogos aleatórios para prever
    jogos_para_prever = []
    num_jogos = 10
    for _ in range(num_jogos):
        jogos_para_prever.append(self.gera_lista_aleatoria_de_elementos_nao_repetidos())

    # Converter os jogos para o formato correto para a previsão
    jogos_para_prever_scaled = scaler.transform(jogos_para_prever)

    # Fazer as previsões
    previsoes = best_model.predict(jogos_para_prever_scaled)

    # Criar um DataFrame para exibir os resultados
    resultados = pd.DataFrame({'Jogo': jogos_para_prever, 'Previsão de Acertos': previsoes})

    # Ordenar os jogos pela previsão de acertos (decrescente)
    resultados_ordenados = resultados.sort_values(by='Previsão de Acertos', ascending=False)

    return resultados_ordenados


ltms = loto('mega_sena')
ltms.treinar_modelo()

# ltlf = loto('loto_facil')
# ltlf.aumentar_dados()
