import pandas as pd

# A coluna deletada é uma classe, entào decidi que seria melhor não colocar ela como dados categoricos, assim, a deletei do DataSet

telescope = pd.read_csv('Clusterizador/dados/telescope.csv', sep = ',')

df = telescope.drop(columns=['class'])

# Salve o DataFrame modificado em um novo arquivo
df.to_csv('Clusterizador/dados/telescope_final.csv', index=False)