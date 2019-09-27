from typing import Any, Union, Tuple

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sb
from sklearn.cluster import KMeans
from scipy.stats import zscore

# IMPORTAR CSV E TRANSFORMAR EM DATAFRAME
pd.set_option('mode.chained_assignment', None)
wine_df = pd.read_csv("E:\\Phyton\\winequality-white.csv", sep=';')

## tirando vinhos com qualidade 3 e 9 da amostra
# idf = idf.drop(idf[idf.quality == 3].index)
# idf = idf.drop(idf[idf.quality == 9].index)

# EXIBIR DADOS DE COLUNAS E LINHAS
print('-------------------------------------------------------------------')
print('INFORMAÇÕES DO DATASET :')
print('Numero de Linhas X Colunas {}'.format(wine_df.shape))

# EXIBIR DADOS DE VALORES FALTANTES
print('Quantidade de valores faltantes agrupados por coluna :')
for col in wine_df.columns:
    print('\t%s: %d' % (col, wine_df[col].isna().sum()))
print('*** linhas com valores faltantes serão removidas do dataframe')
print('')

# EXIBIR DADOS DOS TIPOS DE COLUNA
print('-------------------------------------------------------------------')
print('Tipo de dados das colunas {}'.format(wine_df.dtypes))

# EXIBIR DADOS ESTATÍSTICOS SOBRE DATAFRAME:
print('Dados estatísticos sobre o Dataframe')
print(wine_df.describe().transpose())
print('-------------------------------------------------------------------')

# EXIBIR CORRELAÇÃO ENTRE VARIÁVEIS DO DATAFRAME:
pl.figure(figsize=(10, 7))
sb.heatmap(wine_df.corr(),
           annot=True,
           fmt='.2f',
           cmap='Blues')
pl.title('Correlação entre variáveis do Dataframe da qualidade dos vinhos')
pl.autoscale()
pl.show()

# ADICIONAR COLUNA ID COM NUMERO DA LINHA
wine_df['Id'] = np.arange(len(wine_df))

# CRIAR OUTRO DATAFRAME COM 80% DAS INOFRMACOES
print('Separando o Dataframe em partes para treinamento e teste')
trainning_wine_df = wine_df.sample(frac=0.8)
print('Numero de LINHAS X COLUNAS do arquivo de treinamento com 80% {}'.format(trainning_wine_df.shape))

# CRIAR OUTRO DATAFRAME COM 20% DAS INOFRMACOES
ids = trainning_wine_df['Id']
test_wine_df = wine_df[~wine_df.Id.isin(ids)]
print('Numero de LINHAS X COLUNAS do arquivo de teste com 20% {}'.format(test_wine_df.shape))
print('-------------------------------------------------------------------')

# CRIAR OUTRO DATAFRAME EXCLUINDO AS COLUNAS ID E QUALITY
trainning_wine_df_features = trainning_wine_df.drop(["Id", "quality"], axis=1)
# APLICAR ZCORE PARA NORMALIZAR
trainning_wine_df_features = trainning_wine_df_features.apply(zscore)

# CRIAR OUTRO DATAFRAME EXCLUINDO AS COLUNAS ID E QUALITY
test_wine_df_features = test_wine_df.drop(["Id", "quality"], axis=1)
# APLICAR ZCORE PARA NORMALIZAR
test_wine_df_features = test_wine_df_features.apply(zscore)

# EXECUTAR ALGORITMO K-MEANS COM TOTAL DE 7 GRUPOS PARA A BASE DE TREINAMENTO COM 80% DAS TUPLAS
kmeans_model = KMeans(n_clusters=7)
kmeans_model.fit(trainning_wine_df_features)
test_model = trainning_wine_df
test_model['Classes'] = kmeans_model.predict(trainning_wine_df_features)

# EXECUTAR ALGORITMO K-MEANS COM O MODELO TREINADO PARA A BASE DE TESTES COM 20% DAS TUPLAS
test_wine_df['Classes'] = kmeans_model.predict(test_wine_df_features)


# COMPARAR SE K-MEANS ACERTOU O AGRUPAMENTO
def compare(a, b):
    if a - b == 0:
        return 0
    if (a - b == 1) or (a - b == -1):
        return 1
    if (a - b == 2) or (a - b == -2):
        return 2
    if (a - b == 3) or (a - b == -3):
        return 3
    if (a - b == 4) or (a - b == -4):
        return 4
    if (a - b == 5) or (a - b == -5):
        return 5
    if (a - b == 6) or (a - b == -6):
        return 6
    if (a - b == 7) or (a - b == -7):
        return 7
    if (a - b == 8) or (a - b == -8):
        return 8
    if (a - b == 9) or (a - b == -9):
        return 9
    if (a - b == 10) or (a - b == -10):
        return 10
    else:
        return 11


result_list = []


def compare_true_false(a, b):
    if a - b == 0:
        return 1
    else:
        return 0


result_list_true_false = []


def acerto_com_zero(a, b):
    if a - b == 0:
        return 1
    else:
        return 0


result_list_acerto_com_zero = []


def acerto_com_zero_ou_um(a, b):
    if (a - b == 0) or (a - b == 1) or (a - b == -1):
        return 1
    else:
        return 0


result_list_acerto_com_zero_ou_um = []


def acerto_com_zero_ou_um_ou_dois(a, b):
    if (a - b == 0) or (a - b == 1) or (a - b == -1) or (a - b == 2) or (a - b == -2):
        return 1
    else:
        return 0


result_list_acerto_com_zero_ou_um_ou_dois = []


def acerto_com_zero_ou_um_ou_dois_ou_tres(a, b):
    if (a - b == 0) or (a - b == 1) or (a - b == -1) or (a - b == 2) or (a - b == -2) or (a - b == 3) or (a - b == -3):
        return 1
    else:
        return 0


result_list_acerto_com_zero_ou_um_ou_dois_ou_tres = []


def porcentagem(total, valor_obtido):
    return round(((valor_obtido / total) * 100), 2)


# ITERAR DATAFRAME COMPARANDO LINHA POR LINHA E ARMAZENANDO RESULTADO NUM ARRAY
def test_itertuples(test_wine_df):
    for i in test_wine_df.itertuples():
        result_list.append(compare(i.quality, i.Classes))
        result_list_true_false.append(compare_true_false(i.quality, i.Classes))
        result_list_acerto_com_zero.append(acerto_com_zero(i.quality, i.Classes))
        result_list_acerto_com_zero_ou_um.append(acerto_com_zero_ou_um(i.quality, i.Classes))
        result_list_acerto_com_zero_ou_um_ou_dois.append(acerto_com_zero_ou_um_ou_dois(i.quality, i.Classes))
        result_list_acerto_com_zero_ou_um_ou_dois_ou_tres.append(acerto_com_zero_ou_um_ou_dois_ou_tres(i.quality, i.Classes))


test_itertuples(test_wine_df)

test_wine_df['V_F'] = result_list_true_false
test_wine_df['Zero'] = result_list_acerto_com_zero
test_wine_df['0-1'] = result_list_acerto_com_zero_ou_um
test_wine_df['0-1-2'] = result_list_acerto_com_zero_ou_um_ou_dois
test_wine_df['0-1-2-3'] = result_list_acerto_com_zero_ou_um_ou_dois_ou_tres


# EXIBIR QUALIDADES DOS VINHOS TESTADOS
wine_df.quality.value_counts().plot(kind='pie', autopct='%.2f%%')
pl.axis('equal')
pl.title('Qualidades de vinho no Dataframe Original')
pl.show()

trainning_wine_df.quality.value_counts().plot(kind='pie', autopct='%.2f%%')
pl.axis('equal')
pl.title('Qualidades de vinho no Dataframe de treinamento')
pl.show()

test_wine_df.quality.value_counts().plot(kind='pie', autopct='%.2f%%')
pl.axis('equal')
pl.title('Qualidades de vinho no Dataframe de teste')
pl.show()

print('Qualidades de vinho do dataFrame total')
print(wine_df['quality'].value_counts())
print('')
print('Qualidades de vinho do dataFrame de treinamento')
print(trainning_wine_df['quality'].value_counts())
print('')
print('Qualidades de vinho do dataFrame de teste')
print(test_wine_df['quality'].value_counts())

print('Quantidade de vinhos agrupados por nota de qualidade')
print(wine_df['quality'].value_counts())

# CALCULO DE ACURACIA
print('-------------------------------------------------------------------------------------------------------')
print('Numero de acertos do total de {0} vinhos :  {1} - {2}%'.format(len(test_wine_df), result_list_true_false.count(1),
                                                                      porcentagem(len(test_wine_df),
                                                                                  result_list_true_false.count(1))))
print('Numero de acertos com zero : {0} - {1}%'.format(result_list_acerto_com_zero.count(1), porcentagem(len(test_wine_df),
                                                                                                         result_list_acerto_com_zero.count(
                                                                                                             1))))
print('Numero de acertos com zero ou um : {0} - {1}%'.format(result_list_acerto_com_zero_ou_um.count(1),
                                                             porcentagem(len(test_wine_df),
                                                                         result_list_acerto_com_zero_ou_um.count(1))))
print('Numero de acertos com zero ou um ou dois : {0} - {1}%'.format(result_list_acerto_com_zero_ou_um_ou_dois.count(1),
                                                                     porcentagem(len(test_wine_df),
                                                                                 result_list_acerto_com_zero_ou_um_ou_dois.count(
                                                                                     1))))
print('Numero de acertos com zero ou um ou dois ou tres : {0} - {1}%'.format(
    result_list_acerto_com_zero_ou_um_ou_dois_ou_tres.count(1),
    porcentagem(len(test_wine_df), result_list_acerto_com_zero_ou_um_ou_dois_ou_tres.count(1))))
print('-------------------------------------------------------------------------------------------------------')

# MATRIZ DE CONFUSAO
print('Matriz de Confusão')
valores_reais = pd.Series(test_wine_df['quality'], name='Actual')
valores_preditos = pd.Series(test_wine_df['Classes'], name='Predicted')
df_confusion = pd.crosstab(valores_reais, valores_preditos)

print(df_confusion)
print('-------------------------------------------------------------------------------------------------------')

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=pl.cm.cool):
    pl.matshow(df_confusion, cmap=cmap)
    pl.title(title)
    pl.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    pl.xticks(tick_marks, df_confusion.columns, rotation=45)
    pl.yticks(tick_marks, df_confusion.index)
    pl.ylabel(df_confusion.index.name)
    pl.xlabel(df_confusion.columns.name)
    pl.autoscale()
    pl.show()


plot_confusion_matrix(df_confusion)
