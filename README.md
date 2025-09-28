# **MVP 1 - Algoritmo de predição de custos de seguro saúde**

---



**Gabriel Borges da Conceição e Guilherme Missaggia Bertolo**

**29/09/2025**



---





## Definição do problema

---



O dataset contém informações de 1338 indivíduos, incluindo variáveis demográficas e de saúde, como idade, sexo, índice de massa corporal (IMC), número de filhos, hábito de fumar e região de residência. O objetivo é analisar como esses fatores influenciam o custo do seguro saúde e desenvolver um modelo capaz de prever o valor das taxas de seguro para novos indivíduos.

O desafio consiste em estimar o valor dessas taxas a partir das características fornecidas, analisando como cada fator influencia no custo final. Esse cenário foi formalizado como um problema de regressão supervisionada utilizando técnicas de Machine Learning.

##Premissas
Assume-se que existe uma relação sistemática entre as variáveis independentes e a variável alvo, de forma que os padrões podem ser aprendidos por um modelo supervisionado. Também se considera que os dados são de qualidade suficiente, representativos da população e sem vieses que comprometam sua utilização, além de que os registros são independentes entre si, ou seja, o custo de seguro de um indivíduo não influencia diretamente o de outro. Outro ponto assumido é que as variáveis disponíveis no conjunto de dados são relevantes e suficientes para explicar uma parte significativa da variabilidade nos custos, e que essas relações permanecem estáveis ao longo do tempo, possibilitando a generalização do modelo.

##Dataset


*   Fonte: Kaggle
*   Link: https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset
*   Atributos: O dataset possui 6 variáveis independentes (idade, sexo, IMC, dependentes, htabagismo e região demográfica) e uma variável dependente custo do seguro.


## **1. Importando bibliotecas**

---
! pip install lazypredict

```
# Manipulação de dados
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime

# Visualização
import seaborn as sns
from matplotlib import pyplot as plt

# Leitura de arquivos
import csv
from pandas import read_excel

# Google Colab
from google.colab import drive

# Modelagem
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression

# LazyPredict
from lazypredict.Supervised import LazyRegressor, LazyClassifier

```

## **2. Lendo o dataset**

---



O dataset é lido no Google Drive. No caminho Meu Drive/insurance.


```
drive.mount('/content/gdrive') # Montar arquivos do Google Drive

# Especificando o local onde os dados estão salvos
proj_path = '/content/gdrive/MyDrive/insurance/'

#Especificando dados sobre o arquivo
ac_sheet = 'insurance'
file_name1 = 'insurance.xlsx'
df = read_excel(proj_path + file_name1, sheet_name = ac_sheet)
#df1 = df1.sample(frac=0.3, random_state=42)
n_row1 = df.shape[0] # Numero de linhas
print('Numero Registros:',n_row1)
df
```


## **3. Preparação dos dados**

---



Para a aplicação dos algoritmos de aprendizado de máquina, foi necessário realizar a etapa de pré-processamento, garantindo que todas as variáveis estivessem em formato numérico e adequadas para o treinamento dos modelos.
Os números inteiros foram transformados em números reais e as variáveis categóricas também foram convertidas para o tipo float, assegurando uniformidade no conjunto de dados. Por fim, foram removidas as linhas contendo valores ausentes.

```
df = df.astype({'age':'float'})
df = df.astype({'bmi':'float'})
df = df.astype({'children':'float'})
df['sex'] = df['sex'].map({'female': 2, 'male': 1}).astype(float)
df['smoker'] = df['smoker'].map({'no': 2, 'yes': 1}).astype(float)
df['region'] = df['region'].map({
    'southwest': 1,
    'southeast': 2,
    'northwest': 3,
    'northeast': 4
}).astype(float)

df = df.dropna()
df
```

### **4. Separando as variáveis independentes (X) da variável alvo (Y)**

---


Todas as colunas do DataFrame, exceto charges, foram atribuídas à variável X, representando os atributos que servirão como entrada para o modelo. A coluna charges foi separada na variável Y, correspondendo à variável que se deseja prever.

```
X = df.drop('charges', axis=1)
Y = df.charges
X
```

## **5. Selecionando melhores atributos**

---



Esse código aplica uma técnica de seleção de atributos para identificar quais variáveis têm maior influência sobre a variável alvo (custo do seguro saúde).

O código está comentado porque não obtivemos um resultado vantajoso reduzindo o número de atributos.

```
#names=list(X.columns)
## Seleção de atributos
#n_selected=4 # Digitar aqui o número de atributos a serem selecionados
#test = SelectKBest(score_func=f_regression, k=n_selected)
#fit = test.fit(X, Y)
## summarize scores
#np.set_printoptions(precision=3)
#features = fit.transform(X)
## list(data) or
#performance_list = pd.DataFrame(
#    {'Attribute': names,
#     'Value': fit.scores_
#    })
#performance_list=performance_list.sort_values(by=['Value'], ascending=False)
#names_selected=performance_list.values[0:n_selected,0]

#XX = pd.DataFrame (X, columns = names_selected)
#X=XX
#X
```

## **6. Dividindo dataset em treino e teste**
---


O código separa os dados em treino (70%) e teste (30%) para que o modelo possa ser treinado em uma parte dos dados e avaliado em dados que ele ainda não conhece.

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
```

## **7. Compare ML algoritmos usando lazyregressor**

---



O código automatiza a comparação de múltiplos modelos de regressão, fornecendo rapidamente métricas de desempenho (R², RMSE, tempo de treino), permitindo identificar quais algoritmos funcionam melhor para o dataset.

```
# Defines and builds the lazyregressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, Y_train, Y_test)

print(models)
```

## **8. Otimizando o melhor modelo usando o algoritmo Random Forest Regressor**

---



O algoritmo testa várias combinações de parâmetros do modelo (hiperparâmetros) e, para cada combinação, avaliar seu desempenho de forma confiável usando validação cruzada, escolhendo no final a configuração que obtém os melhores resultados.

```
# Parâmetros para tunar
param = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2] }

# Definir o GridSearchCV com RandomForestRegressor
model = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Treinar o modelo
model.fit(X, Y)

# Previsões no mesmo conjunto
y_pred = model.predict(X)

# R²
r2 = r2_score(Y, y_pred)

# R² Ajustado
n = X.shape[0] # número de observações
p = X.shape[1] # número de features
r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# RMSE
rmse = np.sqrt(mean_squared_error(Y, y_pred))

# Imprimir resultados
print(f"R²: {r2:.4f}")
print(f"R² Ajustado: {r2_ajustado:.4f}")
print(f"RMSE: {rmse:.4f}")
```

## **9. Otimizando o melhor modelo usando o algoritmo Gradient Boosting Regressor**

---



O algoritmo testa várias combinações de parâmetros do modelo (hiperparâmetros) e, para cada combinação, avaliar seu desempenho de forma confiável usando validação cruzada, escolhendo no final a configuração que obtém os melhores resultados.

```
# Parâmetros para tunar
param = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Definir o GridSearchCV com GradientBoostingRegressor
model = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Treinar o modelo
model.fit(X, Y)

# Previsões no mesmo conjunto
y_pred = model.predict(X)

# R²
r2 = r2_score(Y, y_pred)

# R² Ajustado
n = X.shape[0]  # número de observações
p = X.shape[1]  # número de features
r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# RMSE
rmse = np.sqrt(mean_squared_error(Y, y_pred))

# Imprimir resultados
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {r2_ajustado:.4f}")
print(f"RMSE: {rmse:.4f}")
```

## **10. Conclusão**

---



O estudo demonstrou que variáveis como idade, IMC, número de dependentes, hábito de fumar, sexo e região exercem impacto relevante nos custos do seguro saúde. A partir da aplicação de modelos de aprendizado de máquina, buscou-se identificar quais técnicas apresentariam melhor capacidade de generalização.

Nos primeiros testes, modelos como Poisson Regressor, Gradient Boosting Regressor e Random Forest Regressor apresentaram baixo poder explicativo, com valores de R² entre 0,21 e 0,30. Posteriormente, ao aplicar técnicas de otimização, foi obtido um resultado expressivamente superior, com R² de 0,7177 e R² ajustado de 0,7165, utilizando o algoritmo Random Forest Regressor, explicando cerca de 71% da variabilidade dos custos. No entanto, nem todos os ajustes levaram a melhorias consistentes: utilizando o algoritmo Gradient Boosting Regressor, o modelo alcançou R² de 0,4963 (ajustado 0,4940), desempenho intermediário entre os experimentos.

Esses resultados evidenciam que a escolha do algoritmo e de sua parametrização é determinante para a qualidade das previsões. Embora o melhor modelo tenha alcançado desempenho satisfatório, ainda há espaço para aprimoramentos, especialmente considerando o RMSE elevado, que indica grande dispersão nos valores previstos.

Conclui-se que, apesar de limitações como a ausência de variáveis clínicas detalhadas e a suposição de independência entre registros, a modelagem proposta mostra potencial para apoiar processos de precificação de seguros de saúde.

Para trabalhos futuros, recomenda-se o uso de bases mais completas, bem como a avaliação de modelos mais robustos, como redes neurais profundas ou ensembles avançados, visando reduzir erros e aumentar a capacidade preditiva.
