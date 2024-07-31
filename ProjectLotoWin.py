import itertools
import random
import numpy as np
import pandas as pd
import statistics as stat
from itertools import *
from collections import Counter
from tqdm import tqdm
import copy

class lotofacil:

   def __init__(self):
      self.ini = 1
      self.qtd = 15
      self.fim = 25
      self.jlist = []
 
   def gera_lista_aleatoria_de_elementos_nao_repetidos(self):
      unique_ints = set()
      while len(unique_ints) < self.qtd:
        unique_ints.add(random.randint(self.ini, self.fim))
      return list(unique_ints)
 
   def jogos_com_mais_de_n_pontos_em_comum(self, j1, j2, n):
      setj1 = set(j1)
      setj2 = set(j2)
      if len(setj1.intersection(setj2)) >= n:
        return True
      else:
        return False
 
   def lista_jogos_com_max_n_pontos_em_comum(self):
      j = self.gera_lista_aleatoria_de_elementos_nao_repetidos()
      self.jlist = []
      self.jlist.append(j)
      for iind in range(1,11):
        jaux = self.gera_lista_aleatoria_de_elementos_nao_repetidos()
        jbol = []
        jbol.append(self.jogos_com_mais_de_n_pontos_em_comum(self.jlist[0],jaux,10))
        while any(jbol):
          jaux = self.gera_lista_aleatoria_de_elementos_nao_repetidos()
          jbol = []
          for jind in range(0,iind):
            jbol.append(self.jogos_com_mais_de_n_pontos_em_comum(self.jlist[jind],jaux,10))
        self.jlist.append(jaux)
      return self.jlist
 
   def conta_repeticoes(self):
      listtoarray = np.array(self.jlist)
      n = 0
      for ind, val in enumerate(listtoarray):
        setj1 = set(val)
        for j in range(0,ind):
          setj2 = set(listtoarray[j])
          n = max(n,len(setj1.intersection(setj2)))
          print(ind,j,len(setj1.intersection(setj2)))
        print(15*'****')
      return n
 
   def conta_repeticoes_entre_listas(self, lista1, lista2):
      n = 0
      conta = 0
      for jogo2 in lista2:
        setj1 = set(jogo2)
        for jogo1 in lista1:
          setj2 = set(jogo1)
          n = max(n,len(setj1.intersection(setj2)))
          if n > 10:
            cont += 1
            continue
        print(15*'****')
      return n, cont
 
   def gerar_varias_lista_jogos(self):
      for i in range(0,11):
        self.lista_jogos_com_max_n_pontos_em_comum()
        n = self.conta_repeticoes()
        while n > 10:
          self.lista_jogos_com_max_n_pontos_em_comum()
          n = self.conta_repeticoes()
        self.testa_lista_jogos_com_max_n_pontos_em_comum()
        print('+*+*+*+*+*+*+*+*+*+*+*+')
        print(n)
 
   def testa_lista_jogos_com_max_n_pontos_em_comum(self):
      self.lista_jogos_com_max_n_pontos_em_comum()
      listtoarray = np.array(self.jlist)
      n = np.size(listtoarray)
      for ind,val in enumerate(listtoarray):
        setj1 = set(val)
        for j in range(0,ind):
          setj2 = set(listtoarray[j])
          if len(setj1.intersection(setj2)) > 10:
            # print('>>>>>>>',ind,j,val,listtoarray[j])
            print('---',ind,j,len(setj1.intersection(setj2)))
        # print(15*'****')
 
   def gerar_jogo_lotofacil(self):
      return sorted(random.sample(range(1, 26), 15))
 
   def jogos_sao_validos(self,jogos, novo_jogo, max_comum):
      for jogo in jogos:
          if len(set(jogo).intersection(set(novo_jogo))) > max_comum:
              return False
      return True
 
   def gerar_11_jogos(self):
      jogos = []
      tt = 0
      while len(jogos) < 11:
          novo_jogo = self.gerar_jogo_lotofacil()
          if self.jogos_sao_validos(jogos, novo_jogo, 9):
              tt += 1
              #print(tt,novo_jogo)
              jogos.append(novo_jogo)
      self.jlist = jogos
      return jogos
 
   def gerar_combinacoes_lotofacil(self):
      # Gera todas as combinações de 15 números a partir de 25
      self.combinacoes = itertools.combinations(range(1, 26), 15)
      return self.combinacoes
 
   def salvar_jogos_em_arquivo(self, nome_arquivo):
      with open(nome_arquivo, 'w') as f:
        for jogo in self.jlist:
          f.write(str(jogo)+'\n')
 
   def testando_teoria(self):
      totalcombinacoes = 3268760
      dife = 10000
      dife_old = 6268760
      while dife > 10:
         print(20*'##')
         total = 0
         self.gerar_11_jogos()
         self.gerar_combinacoes_lotofacil()
         progress_bar = tqdm(total=totalcombinacoes, desc=" *** ")
         for ind_comb, combinacao in enumerate(self.combinacoes):
            ind_aux = ind_comb
            linha = list(map(int, combinacao))
            progress_bar.update(1)
            for ind_jog, jogo in enumerate(self.jlist):
              n = len(set(jogo).intersection(set(linha)))
              if n > 10:
                total += 1
                ind_aux += 1
                break
 
         progress_bar.close()
         print('Total:', total)
         print('Total de combinacoes:', totalcombinacoes)
         print('Diferenca: ', totalcombinacoes-total)
         dife = totalcombinacoes-total
         if dife < dife_old:
            print('Saving...')
            self.salvar_jogos_em_arquivo('jogos_'+str(dife)+'.txt')
            dife_old = copy.deepcopy(dife)

def main():

	lot = lotofacil()
	lot.testando_teoria()

if __name__ == "__main__":
    main()
