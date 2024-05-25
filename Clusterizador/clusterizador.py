import pandas as pd
from sklearn.cluster import KMeans #Clusterizador
import matplotlib.pyplot as plt #Para gráficos
import math #Matemática
from scipy.spatial.distance import cdist #para calcular as distâncias e distorções
import numpy as np #Para procedimentos numéricos
from sklearn.preprocessing import  MinMaxScaler #Classe normalizadora
from pickle import dump


pd.set_option('display.max_columns', None)

telescope = pd.read_csv('Clusterizador/dados/telescope_final.csv', sep = ',')
dados_numericos = telescope

#1 Normalizar os dados

normalizador = MinMaxScaler()
neo_normalizador = normalizador.fit(dados_numericos) #o método 

dados_numericos_normalizados = normalizador.fit_transform(dados_numericos)
dados_finais = pd.DataFrame(dados_numericos_normalizados)

dump(neo_normalizador, open('Clusterizador/dados/normalizador.pkl','wb'))

distortions = []
K = range(1, 200)
dados_finais.columns = dados_finais.columns.astype(str)

for i in K:
    telescope_kmeans_model = KMeans(n_clusters = i).fit(dados_finais)
    distortions.append(sum(np.min(cdist(dados_finais, telescope_kmeans_model.cluster_centers_, 'euclidean'), axis = 1)/dados_finais.shape[0]))


fig, ax = plt.subplots()
ax.plot(K, distortions)
ax.set(xlabel = 'n Clusters', ylabel = 'Distorção', title = 'Elbow pela distorção')
ax.grid()
fig.savefig('Clusterizador/dados/elbow_distorcao.png')
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

dump(telescope_kmeans_model, open("Clusterizador/dados/telescope_cluster.pkl", "wb"))