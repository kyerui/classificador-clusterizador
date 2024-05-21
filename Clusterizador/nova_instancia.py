import pandas as pd
from pickle import load

nova_instancia = [36.7998,26.0361,9.5629,0.3918,2.1368,2.7994,5.311,-2.2957,20.052,32.8878,'g']
nova_instancia_ds = pd.DataFrame([nova_instancia],  columns=['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class'])

telescope_clusters_kmeans = load(open('Clusterizador/dados/telescope_cluster.pkl', "rb"))
normalizador = load(open("Clusterizador/dados/normalizador.pkl", "rb"))
colunas_categoricas = load(open("Clusterizador/dados/colunas_Categoricas.pkl", "rb"))


dados_categoricos_normalizados = pd.get_dummies(data=nova_instancia_ds[['class']], dtype=int)
dados_numericos = nova_instancia_ds.drop(columns=['class'])
dados_numericos_normalizados = normalizador.transform(dados_numericos)
dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados)

dados_categoricos = pd.DataFrame(columns=colunas_categoricas)


dados_categoricos = dados_categoricos.apply(lambda x: pd.Series([pd.NA]*len(x), index=x), axis=1)

# Concatenar os DataFrames
dados_completos = pd.concat([dados_categoricos, dados_categoricos_normalizados], axis=0)

dados_completos = dados_completos[dados_categoricos.columns]
# Substituir os valores NaN por pd.NA
dados_completos = dados_completos.fillna(0)


dados_completos = dados_numericos_normalizados.join(dados_completos, how='left')

dados_completos.columns = dados_completos.columns.astype(str)

print(f"A nova instância pertence ao grupo: {telescope_clusters_kmeans.predict(dados_completos.values)}")
print("\nO grupo ao qual esta nova instância pertence é caracterizado a partir do: \nEixo principal da elipse.\nEixo menor da elipse.\nLog de 10 da soma do conteúdo de todos os pixels(tamanho)\nProporção da soma dos dois pixels mais altos sobre o tamanho.\nProporção do pixel mais alto sobre o tamanho.\nDistância do pixel mais alto ao centro, projetada no eixo principal.\n3ª raiz do terceiro momento ao longo do eixo maior.\n3ª raiz do terceiro momento ao longo do eixo menor.\nÂngulo do eixo principal com o vetor até a origem.\nDistância da origem ao centro da elipse.\nPor fim a classe")


print(f"\nA centroide deste grupo está localizada em: \n{telescope_clusters_kmeans.cluster_centers_[telescope_clusters_kmeans.predict(dados_completos)]}")
                    