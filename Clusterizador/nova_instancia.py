import pandas as pd
from pickle import load

nova_instancia = [23.8172,9.5728,2.3385,0.6147,0.3922,27.2107,-6.4633,-7.1513,10.449,116.737]
nova_instancia_ds = pd.DataFrame([nova_instancia],  columns=['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist'])

telescope_clusters_kmeans = load(open('Clusterizador/dados/telescope_cluster.pkl', "rb"))
normalizador = load(open("Clusterizador/dados/normalizador.pkl", "rb"))


dados_numericos_normalizados = normalizador.transform(nova_instancia_ds)
dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns=['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist'])

print(dados_numericos_normalizados)

print(f"A nova instância pertence ao grupo: {telescope_clusters_kmeans.predict(dados_numericos_normalizados)}")

 
centroide = telescope_clusters_kmeans.cluster_centers_[telescope_clusters_kmeans.predict(dados_numericos_normalizados)]
print(centroide)

centroide_list = list(centroide[0])
print(centroide_list)

centroide_num = centroide_list[:10]

# centroide_num = pd.DataFrame({0: centroide_num})
# # Transpor o DataFrame
# df_transposed = centroide_num.transpose()
# dados_norm_legiveis_num = normalizador.inverse_transform(df_transposed)

# print(dados_norm_legiveis_num)




                    