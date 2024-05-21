import pandas as pd
from sklearn.cluster import KMeans #Clusterizador
import matplotlib.pyplot as plt #Para gráficos
import math #Matemática
from scipy.spatial.distance import cdist #para calcular as distâncias e distorções
import numpy as np #Para procedimentos numéricos
from sklearn.preprocessing import  MinMaxScaler #Classe normalizadora
from pickle import dump

pd.set_option('display.max_columns', None)

telescope = pd.read_csv('telescope.csv', sep = ',')

#1 Normalizar os dados
#1.1 Segmentar dados numéricos e dados categóricos
dados_numericos = telescope.drop(columns=['class'])
dados_categoricos = telescope[['class']]
dados_categoricos_normalizados = pd.get_dummies(data= dados_categoricos, dtype='int16')

colunas_categoricas = dados_categoricos_normalizados.columns
with open("colunas_categoricas.pkl", "wb") as f:
    dump(colunas_categoricas, f)
    
    
#1.2 Gerar o modelo normalizador para uso posterior
normalizador = MinMaxScaler()
neo_normalizador = normalizador.fit(dados_numericos) #o método fit() gera o modelo para normalização

dados_numericos_normalizados = normalizador.fit_transform(dados_numericos) #o método fit_transform gera os dados normalizados

#1.4. Recompor os dados na forma de data frames
#Incorporar os dados normalizados em um único objeto
#converter o ndarray em data frame

dados_finais = pd.DataFrame(dados_numericos_normalizados)
dados_finais = dados_finais.join(dados_categoricos_normalizados, how = 'left')

dump(neo_normalizador, open('normalizador.pkl','wb'))



distortions = []
K = range(1, 101)
dados_finais.columns = dados_finais.columns.astype(str)

for i in K:
    telescope_kmeans_model = KMeans(n_clusters = i).fit(dados_finais)
    distortions.append(sum(np.min(cdist(dados_finais, telescope_kmeans_model.cluster_centers_, 'euclidean'), axis = 1)/dados_finais.shape[0]))


fig, ax = plt.subplots()
ax.plot(K, distortions)
ax.set(xlabel = 'n Clusters', ylabel = 'Distorção', title = 'Elbow pela distorção')
ax.grid()
fig.savefig('elbow_distorcao.png')
plt.show()



# Calcular o número ótimo de clusters
x0 = K[0]
y0 = distortions[0]
xn = K[len(K) - 1]
yn = distortions[len(distortions)-1]
# Iterar nos pontos gerados durante os treinamentos preliminares
distancias = []
for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)

# Maior distância
n_clusters_otimo = K[distancias.index(np.max(distancias))]

telescope_kmeans_model = KMeans(n_clusters = n_clusters_otimo, random_state=42).fit(dados_finais)

dump(telescope_kmeans_model, open("telescope_cluster.pkl", "wb"))