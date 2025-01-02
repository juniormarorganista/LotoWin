import glob
import random
import itertools
import numpy as np
import pandas as pd
import statistics as stat
import re
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
      self.loteria = loteria
      self.ini = 1
      self.qtd_jogos = 10
      self.read_results()

      if self.loteria == 'mega_sena':
         self.fim = 60
         self.num_sort = 6
      
         self.df['acertos'] = 6
         self.qtd_num_int = 4

      elif self.loteria == 'loto_facil':
         self.fim = 25
         self.num_sort = 15

         self.df['acertos'] = 15
         self.qtd_num_int = 10

      elif self.loteria == 'quina':
         self.fim = 80
         self.num_sort = 5

         self.df['acertos'] = 5

      else:
         print('Loteria não encontrada')

      self.df = self.df.drop(columns=['Concurso', 'Data'])

   def read_results(self):
      
      arquivo = self.loteria+'_*'
      aux = arquivo.replace("_"," ").replace("*","")
      print(f'Lendo os dados dos resultados anteriores da {aux}')
      
      # Localizar o arquivo
      arquivos = glob.glob(f"{arquivo}.xlsx")
      if not arquivos:
         raise FileNotFoundError(f"Nenhum arquivo encontrado com o padrão: {arquivo}!!!")

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
      
      print(f'Aumentando os dados e comparando com o resultados originais!!!')
      self.aumentar_dados()

      X = self.df.drop('acertos', axis=1)
      y = self.df['acertos']

      # Garantir que X seja um DataFrame com nomes das colunas consistentes
      X = pd.DataFrame(X, columns=self.df.columns[:-1])
      
      # Feature scaling com StandardScaler
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)  # Escalar X com os nomes consistentes

      # Split data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

      # Dicionário de modelos
      models = {
         'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators = 100, max_depth = 20, min_samples_split = 5),
         'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42, n_estimators = 100, learning_rate = 0.1, max_depth = 5)
         # 'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),  # Ajustado para convergência
         # 'SVC': SVC(probability=True, random_state=42),  # Support Vector Classifier com probabilidade
         # 'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
         # 'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),  # K-Nearest Neighbors com 5 vizinhos
         # 'GaussianNB': GaussianNB(),  # Naive Bayes
         # 'AdaBoostClassifier': AdaBoostClassifier(random_state=42, algorithm="SAMME")  # AdaBoost com estimadores fracos
      }

      results = {}
      
      # Avaliar todos os modelos
      for name, model in models.items():
         print(f"Testando o {name}!!!")
         model.fit(X_train, y_train)
         y_pred = model.predict(X_test)
         accuracy = accuracy_score(y_test, y_pred)
         results[name] = accuracy
         print(f"{name} obteve Accuracy: {accuracy}")
         print(20*'*+-')

      # Encontrar o melhor modelo
      best_model_name = max(results, key=results.get)
      best_model = models[best_model_name]
      print(f"\nBest Model: {best_model_name}")

      # Treinar o melhor modelo
      best_model.fit(X_train, y_train)


      # Lista para armazenar os jogos que atendem ao critério
      jogos_selecionados = []
      quantidade_jogos = 60
      valor_desejado = 6
      # Gerar jogos até atingir a quantidade desejada
      while len(jogos_selecionados) < quantidade_jogos:
         # Gerar um novo jogo aleatório
         jogo = self.gera_lista_aleatoria_de_elementos_nao_repetidos()

         # Prever a quantidade de acertos do jogo
         jogo_df = pd.DataFrame([jogo], columns=self.df.columns[:-1])
         jogo_escalado = scaler.transform(jogo_df)  # Escalar o jogo
         previsao = best_model.predict(jogo_escalado)[0]

         # Verificar se a previsão atende ao critério
         if previsao == valor_desejado:
               jogos_selecionados.append(jogo.sort())
               print(f"Jogo {len(jogos_selecionados)}: {jogo}, Previsão: {previsao}")

      print(f"\nTotal de jogos selecionados: {len(jogos_selecionados)}")
      return jogos_selecionados

      # jogos_para_prever = []
      # qtd_jog = 0
      # jogos_para_prever_aux = self.gera_lista_aleatoria_de_elementos_nao_repetidos()
      # while qtd_jog < 7:
         
      #    jogos_para_prever_aux = self.gera_lista_aleatoria_de_elementos_nao_repetidos()
      #    jogos_para_prever_df_aux = pd.DataFrame(jogos_para_prever_aux, columns=self.df.columns[:-1])
      #    jogos_para_prever_scaled_aux = scaler.transform(jogos_para_prever_df_aux)
      #    previsoes = best_model.predict(jogos_para_prever_scaled_aux)
      #    print(previsoes)
      #    # if previsoes > 5:
      #    #    print()
      #    #    qtd_jog += 1 

      # # Gerar jogos aleatórios para prever
      # jogos_para_prever = []
      # num_jogos = 6
      # for _ in range(num_jogos):
      #    jogos_para_prever.append(self.gera_lista_aleatoria_de_elementos_nao_repetidos())

      # # Converter os jogos gerados para o formato correto com os nomes das colunas
      # jogos_para_prever_df = pd.DataFrame(jogos_para_prever, columns=self.df.columns[:-1])

      # # Escalar os jogos para prever
      # jogos_para_prever_scaled = scaler.transform(jogos_para_prever_df)

      # # Fazer as previsões
      # previsoes = best_model.predict(jogos_para_prever_scaled)
      # print(previsoes > 5)
      # # Criar um DataFrame para exibir os resultados
      # resultados = pd.DataFrame({'Jogo': jogos_para_prever, 'Previsão de Acertos': previsoes})

      # # Ordenar os jogos pela previsão de acertos (decrescente)
      # resultados_ordenados = resultados.sort_values(by='Previsão de Acertos', ascending=False)

      # return resultados_ordenados
   
   def treinar_modelo_com_hiperparametros(self):
      print(f'Aumentando os dados e comparando com os resultados originais!!!')
      self.aumentar_dados()

      # Separar features (X) e alvo (y)
      X = self.df.drop('acertos', axis=1)
      y = self.df['acertos']

      # Feature scaling
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)

      # Split data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

      # Definições de modelos e hiperparâmetros
      models_with_params = {
         'RandomForestClassifier': {
               'model': RandomForestClassifier(random_state=42),
               'params': {
                  'n_estimators': [50, 100, 150],
                  'max_depth': [None, 10, 20],
                  'min_samples_split': [2, 5]
               }
         },
         'GradientBoostingClassifier': {
               'model': GradientBoostingClassifier(random_state=42),
               'params': {
                  'n_estimators': [50, 100],
                  'learning_rate': [0.1, 0.01],
                  'max_depth': [3, 5]
               }
         },
         'LogisticRegression': {
               'model': LogisticRegression(random_state=42, max_iter=1000),
               'params': {
                  'C': [0.1, 1, 10],
                  'solver': ['lbfgs', 'liblinear']
               }
         },
         'SVC': {
               'model': SVC(probability=True, random_state=42),
               'params': {
                  'C': [0.1, 1, 10],
                  'kernel': ['linear', 'rbf']
               }
         },
         'DecisionTreeClassifier': {
               'model': DecisionTreeClassifier(random_state=42),
               'params': {
                  'max_depth': [None, 10, 20],
                  'min_samples_split': [2, 5]
               }
         },
         'KNeighborsClassifier': {
               'model': KNeighborsClassifier(),
               'params': {
                  'n_neighbors': [3, 5, 7],
                  'weights': ['uniform', 'distance']
               }
         },
         'GaussianNB': {
               'model': GaussianNB(),
               'params': {}
         },
         'AdaBoostClassifier': {
               'model': AdaBoostClassifier(random_state=42, algorithm="SAMME"),
               'params': {
                  'n_estimators': [50, 100],
                  'learning_rate': [1.0, 0.5]
               }
         }
      }

      results = {}

      # Testar modelos e combinações de hiperparâmetros
      for name, config in models_with_params.items():
         print(f"Testando o {name}")
         model = config['model']
         param_grid = config['params']

         if param_grid:  # Se o modelo tem hiperparâmetros para testar
               best_score = 0
               best_params = None

               # Criar combinações de parâmetros
               param_combinations = list(product(*param_grid.values()))
               for param_set in param_combinations:
                  # Atualizar os parâmetros do modelo
                  params = dict(zip(param_grid.keys(), param_set))
                  model.set_params(**params)

                  # Treinar e avaliar
                  model.fit(X_train, y_train)
                  y_pred = model.predict(X_test)
                  accuracy = accuracy_score(y_test, y_pred)

                  # Atualizar o melhor modelo para este classificador
                  if accuracy > best_score:
                     best_score = accuracy
                     best_params = params

               print(f"{name} - Melhor combinação: {best_params} com Accuracy: {best_score:.4f}")
               results[name] = {'accuracy': best_score, 'params': best_params}

         else:  # Se não há hiperparâmetros para testar
               model.fit(X_train, y_train)
               y_pred = model.predict(X_test)
               accuracy = accuracy_score(y_test, y_pred)
               results[name] = {'accuracy': accuracy}
               print(f"{name} - Sem hiperparâmetros, Accuracy: {accuracy:.4f}")

      # Selecionar o melhor modelo geral
      best_model_name = max(results, key=lambda x: results[x]['accuracy'])
      best_model = models_with_params[best_model_name]['model']
      best_params = results[best_model_name]['params']
      print(f"\nMelhor modelo geral: {best_model_name} com parâmetros {best_params}")

      # Treinar o melhor modelo com os melhores parâmetros
      best_model.set_params(**best_params)
      best_model.fit(X_train, y_train)

      return results

if __name__ == '__main__':
   ltms = loto('mega_sena')
   ltms.treinar_modelo()
   # print(ltms.treinar_modelo_com_hiperparametros())

   # ltlf = loto('loto_facil')
   # ltlf.aumentar_dados()