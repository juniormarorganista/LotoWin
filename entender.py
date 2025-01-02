import requests
from bs4 import BeautifulSoup

# URL do site com os resultados
url = "https://asloterias.com.br/lista-de-resultados-da-lotofacil?ordenacao=sorteio"

# Fazer a requisição HTTP
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Erro ao acessar a página: {response.status_code}")

# Analisar o HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Imprimir os primeiros 500 caracteres do HTML para entender a estrutura
print(soup.prettify()[:500])

# Procurar por elementos <strong> (ou outros) que contenham os concursos
tags_strong = soup.find_all('strong')
print("\nPrimeiras tags <strong> encontradas:")
for tag in tags_strong[:5]:
    print(tag.text)

