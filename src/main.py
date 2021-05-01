#Passo 1 - Definir Database

from scipy.stats import shapiro
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets

df = pd.read_csv('data/concrete_data.csv')

#Passo 2 - Limpar Database

#exibir valores ausentes ou null
df.isnull().sum().sort_values(ascending=False)[:10]
print("Número de linhas e colunas no conjunto de treinamento:", df.shape)
attributes = list(df.columns)
#removendo valores nulos
df.dropna()

#preencher os nulos
df.fillna(df.mean(0))

#remover duplicados
df.drop_duplicates()

#Passo 3 - Definir x e y
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(x)
print(y)

#Passo 4 - Média
averageX = np.mean(x)
averageY = np.mean(y)

print("Média de x: " + str(averageX))  #calcula a média de x
print("Média de y: " + str(averageY))  #calcula a média de y

#Passo 5 - Variância
varianceX = np.var(x)
varianceY = np.var(y)

print("Variância de x: " + str(varianceX))
print("Variância de y: " + str(varianceY))

#Passo 6 - Desvio Padrão
deviationX = np.std(x)
deviationY = np.std(y)

print("Desvio Padrão de x: " + str(deviationX))  #calcula o desvio padrão de x
print("Desvio Padrão de y: " + str(deviationY))  #calcula o desvio padrão de y

#Passo 7 - Mediana
medianX = np.median(x)
medianY = np.median(y)

print("Mediana de x: " + str(medianX))  #calcula o desvio padrão de x
print("Mediana de y: " + str(medianY))  #calcula o desvio padrão de y

#Passo 8 - Histograma

# Histograma de x
h = np.histogram(x, bins='auto')  #calcula o histograma
print(x)
plt.hist(y, bins='auto')
plt.title('Dados')
plt.ylabel('Frequência')
plt.xlabel('Valores')
plt.show()

# Histograma de y
h = np.histogram(y, bins='auto')  #calcula o histograma
print(h)
plt.hist(y, bins='auto')
plt.title('Dados')
plt.ylabel('Frequência')
plt.xlabel('Valores')
plt.show()

#Passo 9 - Coeficiente de Correlação
print('\n\n\n\nPearson')
print(df.corr(method='pearson'))

#Passo 10 - Teste de Normalidade
from scipy.stats import shapiro
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets
data = pd.read_csv('data/concrete_data.csv')
print("Número de linhas e colunas:", data.shape)
data.head(25)

# Analisar se a coluna sepal.length tem distribuição normal

data = data.to_numpy()
x = data[:, 0]
# normalidade test
stat, p = shapiro(x)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpretação
alpha = 0.05
if p > alpha:
    print('Amostra Gaussiana (aceita H0)')
else:
    print('Amostra não Gausssiana (rejeita H0)')
# Verificação atrav´s do histograma
plt.hist(x, bins='auto')
plt.title('Dados')
plt.ylabel('Frequência')
plt.xlabel('Valores')
plt.show()

from scipy.stats import shapiro
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets
data = pd.read_csv('data/concrete_data.csv')
print("Número de linhas e colunas:", data.shape)
data.head(25)

# Analisar se a coluna sepal.length tem distribuição normal

data = data.to_numpy()
y = data[0, :]
# normalidade test
stat, p = shapiro(y)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpretação
alpha = 0.05
if p > alpha:
    print('Amostra Gaussiana (aceita H0)')
else:
    print('Amostra não Gausssiana (rejeita H0)')
# Verificação atrav´s do histograma
plt.hist(y, bins='auto')
plt.title('Dados')
plt.ylabel('Frequência')
plt.xlabel('Valores')
plt.show()
