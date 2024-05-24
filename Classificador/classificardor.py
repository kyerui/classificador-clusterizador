
import pandas as pd
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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
dados_finais = pd.concat([dados_atributos_b, dados_classes_b], axis=1)

####################################################################################################################################
dados_finais_train_b = dados_finais.sample(680)
dados_f_classes = dados_finais_train_b['class']
dados_f_atributos = dados_finais_train_b.drop(columns=['class'])


atributos_train, atributos_test, classes_train, classes_test = train_test_split(dados_f_atributos, dados_f_classes)

tree = DecisionTreeClassifier()


# Definir a grade de hiperparâmetros
param_grid = [{

    'criterion' : ['gini', 'entropy', 'log_loss'],
    'splitter' : ['best','random'],
    'max_depth' : [None, 5, 10, 20, 30], 
    'min_samples_split' : [2, 5, 0.5, 0.7] ,
    'min_samples_leaf' : [1, 2,  0.5, 0.7],
    'max_features' : ['auto', 'sqrt', 'log2', None, 0.5, 0.7]
}]

# Implementar o Grid Search com validação cruzada
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=10)
grid_search.fit(atributos_train, classes_train)

# Imprimir os melhores parâmetros e o melhor score
print("Melhores Parâmetros: ", grid_search.best_params_)
print("Melhor Score: ", grid_search.best_score_)

# Avaliar o modelo no conjunto de teste
best_model = grid_search.best_estimator_

# Avaliar o modelo no conjunto de teste
test_accuracy = best_model.score(atributos_test, classes_test)
print("Acurácia no conjunto de teste: ", test_accuracy)


####################################################################################################################################
# AVALIAÇÃO DA ACURÁCIA COM CROSS VALIDATION


tree = DecisionTreeClassifier(criterion=grid_search.best_params_['criterion'], splitter=grid_search.best_params_['splitter'], max_depth=grid_search.best_params_['max_depth'], min_samples_split=grid_search.best_params_['min_samples_split'], min_samples_leaf=grid_search.best_params_['min_samples_leaf'], max_features=grid_search.best_params_['max_features'])

tree.fit(dados_atributos_b, dados_classes_b)

scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(tree,dados_atributos_b,dados_classes_b, cv = 5, scoring = scoring)

print(scores_cross['test_precision_macro'].mean())
print(scores_cross['test_recall_macro'].mean())

score_cross_val = cross_val_score(tree, dados_atributos_b, dados_classes_b, cv = 5)
print(score_cross_val.mean(), ' - ', score_cross_val.std())


# Salvar o modelo para uso posterior
dump(tree, open('Classificador/dados/sdss_tree_model_cross.pkl', 'wb'))







