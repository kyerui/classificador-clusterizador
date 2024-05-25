
import numpy as np
from pickle import load
nova_instancia = [[1.24E+18,2.88E+18,-185.5744857,0.701402405,19.11034,19.10073,18.66402,18.58816,18.6467,109,301,4,157,1242,52901,588,1.329944,1.258982,1.05376,1.10939,1.20618,18.03211,22.02316,34.88033,32.6587,33.41023,0.6582081,0.6382042,0.5459,0.5611792,0.5607421,19.35507,18.67,19.1216,18.60126,18.63812,0.5999954,0.5861196,0.6475893,0.2500816,0.05874168,1.161747]]

normalizador = load(open('Classificador/dados/SDSS_normalizador.pkl','rb'))
sdss_classificador = load(open('Classificador/dados/sdss_tree_model_cross.pkl','rb'))

# Normalizar a nova instância
nova_instancia_normalizada = normalizador.transform(nova_instancia)

# Classificar a nova instância
resultado = sdss_classificador.predict(nova_instancia_normalizada)

# Obter a distribuição probabilística
dist_proba = sdss_classificador.predict_proba(nova_instancia_normalizada)

# Determinar a classe com a maior probabilidade
indice = np.argmax(dist_proba[0])
classe_predita = sdss_classificador.classes_[indice]
score = dist_proba[0][indice]

# Exibir o resultado
print("\n########################################################################\n")
print("Classificado como:", classe_predita)
print('Score:', str(score))
print("Índice da classe predita:", np.argmax(dist_proba[0]))
print("Classes disponíveis no modelo:", sdss_classificador.classes_)
print("\n########################################################################")