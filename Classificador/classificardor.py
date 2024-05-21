
import pandas as pd
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from pickle import dump
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter



dados = pd.read_csv('Classificador/dados/SDSS_DR18.csv', sep=',')

# Segmentar os dados em atributos e classes
dados_atributos = dados.drop(columns=['class'])
dados_classes = dados['class']



# Normalizador

normalizador = MinMaxScaler()
sdss_normalizador = normalizador.fit(dados_atributos) 

# Salva o modelo
dump(sdss_normalizador, open('Classificador/dados/SDSS_normalizador.pkl', 'wb'))

# ####################################################################################################################################

# Normalizar a base de dados para treinamento
dados_atributos_normalizados = normalizador.fit_transform(dados_atributos)

# Recompor em DataFrame
dados_finais = pd.DataFrame(data=dados_atributos_normalizados)
dados_finais = dados_finais.join(dados_classes, how='left')


# ####################################################################################################################################
# Balanceamento
# Frequencia de classes
print('Frequência das classes original: ', dados_classes.value_counts())

# Aplicar SMOTE (técnica que amplia a frequência das classes sintetizando novas instâncias e respeitando a probabilidade do segmento original de dados)

dados_atributos = dados_finais.drop(columns=['class'])
dados_classes = dados_finais['class']

# Construir um objeto a partir do SMOTE
resampler = SMOTE()

# Executo o balanceamento
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

# Verificar a frequência das classes após o balanceamento
print('Frequência de classes após balanceamento:')

classes_count = Counter(dados_classes_b)
print(classes_count)

# Converter os dados balanceados em DataFrames
dados_atributos_b = pd.DataFrame(dados_atributos_b)
dados_classes_b = pd.DataFrame(dados_classes_b)



####################################################################################################################################
# Treinamento | AVALIAÇÃO DA ACURÁCIA COM CROSS VALIDATION



atributos_train, atributos_test, classes_train, classes_test = train_test_split(dados_atributos_b, dados_classes_b)


tree = DecisionTreeClassifier()

# TREINAMENTO E TESTES COM HOLD OUT (70% para treinamento e 30% para testes)
sdds_tree = tree.fit(atributos_train, classes_train)

# pretestar o modelo
Classe_test_predict = sdds_tree.predict(atributos_test)



scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(tree,dados_atributos_b,dados_classes_b, cv = 10, scoring = scoring)

print(scores_cross['test_precision_macro'].mean())
print(scores_cross['test_recall_macro'].mean())


# Treinar o modelo com a base normalizada, balanceada e completa
sdss_tree = tree.fit(dados_atributos_b, dados_classes_b)

# Salvar o modelo para uso posterior
dump(sdss_tree, open('Classificador/dados/sdss_tree_model_cross.pkl', 'wb'))

####################################################################################################################################

print('Acurácia global: ', metrics.accuracy_score(classes_test, Classe_test_predict))

# Matriz de confusão
ConfusionMatrixDisplay.from_estimator(sdss_tree, atributos_test, classes_test)
plt.xticks(rotation=90, ha='right')
plt.show()

####################################################################################################################################

sdss_tree_cross = tree.fit(dados_atributos_b, dados_classes_b)
ConfusionMatrixDisplay.from_estimator(sdss_tree_cross, dados_atributos_b, dados_classes_b)
plt.xticks(rotation=90, ha='right')
plt.show()

dump(sdss_tree_cross, open('Classificador/dados/sdss_tree_cross.pkl', 'wb'))






