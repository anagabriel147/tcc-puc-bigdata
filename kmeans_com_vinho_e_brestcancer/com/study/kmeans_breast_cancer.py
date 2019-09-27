import numpy  as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sb

from sklearn.cluster import KMeans
from scipy.stats import zscore

# IMPORTAR CSV E TRANSFORMAR EM DATAFRAME
pd.set_option('mode.chained_assignment', None)
cancer_df = pd.read_csv("E:\\Phyton\\BreastCancer.csv", sep=',')

# EXIBIR DADOS DE COLUNAS E LINHAS
print('-------------------------------------------------------------------')
print('INFORMAÇÕES DO DATASET :')
print('Numero de Linhas X Colunas {}'.format(cancer_df.shape))


# EXIBIR DADOS DE VALORES FALTANTES
print('Quantidade de valores faltantes agrupados por coluna :')
for col in cancer_df.columns:
    print('\t%s: %d' % (col, cancer_df[col].isna().sum()))
print('*** linhas com valores faltantes serão removidas do dataframe')
print('')

# REMOVER LINHAS DO DATAFRAME COM VALORES NULOS
print('Quantidade de linhas no Dataframe antes da remoção de valores faltantes = %d' % (cancer_df.shape[0]))
cancer_df = cancer_df.replace('NA', np.NaN)
cancer_df = cancer_df.dropna()
print('Quantidade de linhas no Dataframe após o discarte de valores faltantes = %d' % (cancer_df.shape[0]))
print('-------------------------------------------------------------------')

# EXIBIR DADOS DOS TIPOS DE COLUNA
print('-------------------------------------------------------------------')
print('Tipo de dados das colunas {}'.format(cancer_df.dtypes))

# EXIBIR DADOS ESTATÍSTICOS SOBRE DATAFRAME:
print('Dados estatísticos sobre o Dataframe')
print(cancer_df.describe().transpose())
print('-------------------------------------------------------------------')

# EXIBIR CORRELAÇÃO ENTRE VARIÁVEIS DO DATAFRAME:
cancer_df2=cancer_df.drop(["Id"], axis=1)
pl.figure(figsize=(10, 7))
sb.heatmap(cancer_df2.corr(),
           annot=True,
           fmt='.2f',
           cmap='Blues')
pl.title('Correlação entre variáveis do Dataframe de cancer de mama')
pl.autoscale()
pl.show()

# CRIAR OUTRO DATAFRAME COM 80% DAS INOFRMACOES
print('Separando o Dataframe em partes para treinamento e teste')
trainning_cancer_df = cancer_df.sample(frac=0.8)
print('Numero de LINHAS X COLUNAS do arquivo de treinamento com 80% {}'.format(trainning_cancer_df.shape))

# CRIAR OUTRO DATAFRAME COM 20% DAS INOFRMACOES
ids = trainning_cancer_df['Id']
test_cancer_df = cancer_df[~cancer_df.Id.isin(ids)]
print('Numero de LINHAS X COLUNAS do arquivo de teste com 20% {}'.format(test_cancer_df.shape))
print('-------------------------------------------------------------------')

# CRIAR OUTRO DATAFRAME EXCLUINDO AS COLUNAS ID E CLASSE
trainning_cancer_df_features = trainning_cancer_df.drop(["Id", "Class"], axis=1)
# APLICAR ZCORE PARA NORMALIZAR
trainning_cancer_df_features = trainning_cancer_df_features.apply(zscore)

# CRIAR OUTRO DATAFRAME EXCLUINDO AS COLUNAS ID E CLASSE
test_cancer_df_features = test_cancer_df.drop(["Id", "Class"], axis=1)
# APLICAR ZCORE PARA NORMALIZAR
test_cancer_df_features = test_cancer_df_features.apply(zscore)

# EXECUTAR ALGORITMO K-MEANS COM TOTAL DE 2 GRUPOS PARA A BASE DE TREINAMENTO COM 80% DAS TUPLAS
kmeans_model = KMeans(n_clusters=2)
kmeans_model.fit(trainning_cancer_df_features)
test_model = trainning_cancer_df
test_model['Classes'] = kmeans_model.predict(trainning_cancer_df_features)

# EXECUTAR ALGORITMO K-MEANS COM O MODELO TREINADO PARA A BASE DE TESTES COM 20% DAS TUPLAS
test_cancer_df['Classes'] = kmeans_model.predict(test_cancer_df_features)


# COMPARAR SE K-MEANS ACERTOU O AGRUPAMENTO
def compare(a, b):
    if a - b == 0:
        return 0
    else:
        return 1


result_list = []


def compare_true_false(a, b):
    if a - b == 0:
        return 1
    else:
        return 0


result_list_true_false = []


def porcentagem(total, valor_obtido):
    return round(((valor_obtido / total) * 100), 2)


# ITERAR DATAFRAME COMPARANDO LINHA POR LINHA E ARMAZENANDO RESULTADO NUM ARRAY
def test_itertuples(test_cancer_df):
    for i in test_cancer_df.itertuples():
        result_list.append(compare(i.Class, i.Classes))
        result_list_true_false.append(compare_true_false(i.Class, i.Classes))


test_itertuples(test_cancer_df)

test_cancer_df['Acertos'] = result_list_true_false

# EXIBIR TIPOS DE CANCERES TESTADOS ( BENIGNO OU MALIGINO )
cancer_df.Class.value_counts().plot(kind='pie', autopct='%.2f%%')
pl.axis('equal')
pl.title('Tipos de Cancer no Dataframe Original')
pl.show()

trainning_cancer_df.Class.value_counts().plot(kind='pie', autopct='%.2f%%')
pl.axis('equal')
pl.title('Tipos de Cancer no Dataframe de treinamento')
pl.show()

test_cancer_df.Class.value_counts().plot(kind='pie', autopct='%.2f%%')
pl.axis('equal')
pl.title('Tipos de Cancer no Dataframe de teste')
pl.show()

print('Canceres agrupados por benigno ou malígno do dataFrame total')
print(cancer_df['Class'].value_counts())
print('')
print('Canceres agrupados por benigno ou malígno do dataFrame de treinamento')
print(trainning_cancer_df['Class'].value_counts())
print('')
print('Canceres agrupados por benigno ou malígno do dataFrame de teste')
print(test_cancer_df['Class'].value_counts())

# CALCULO DE ACURACIA
print('-------------------------------------------------------------------------------------------------------')
print('Numero de acertos ( Acuracia ) do total de {0} amostras :  {1} - {2}%'.format(len(test_cancer_df),
                                                                                     result_list_true_false.count(1),
                                                                                     porcentagem(len(test_cancer_df),
                                                                                                 result_list_true_false.count(
                                                                                                     1))))
print('-------------------------------------------------------------------------------------------------------')

# MATRIZ DE CONFUSAO
print('Matriz de Confusão')
valores_reais = pd.Series(test_cancer_df['Class'], name='Actual')
valores_preditos = pd.Series(test_cancer_df['Classes'], name='Predicted')
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
