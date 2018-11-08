
# Projeto prático #4 - Modelo para classificação de sementes de trigo

Equipe:

- Lucas Frota
- Wilson neto

# 1. Imports


```python
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import numpy as np

from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from math import ceil

%matplotlib inline
```


```python
import warnings
warnings.simplefilter("ignore")
```

# 2. funções uteis


```python
def cast_tab_to_comma_separated_values(file_name):
    '''
    esta função converte um arquivo com valores separados por tabulação em um dataframe com valores do tipo float
    '''
    datasetlines = open(file_name, "r").readlines()
    df_list = []
    for line in datasetlines:
        line = line.replace("\n", "")
        line = line.split("\t")
        line = [float(i) for i in line if(i != '')]
        df_list.append(line)
       
    return pd.DataFrame(df_list)
```


```python
def plot_mult_histograms(df, columns, has_legend=True):
    '''
    esta função mostra um histograma com determinadas colunas de um dataframe
    '''
    for column in columns:
        plt.hist(df[column], label=column)
        
    if(has_legend):
        plt.legend()
    plt.show()
```


```python
def plot_combination(df, x, y):
    '''
    esta função mostra um grafico com os pontos coloridos de acordo com a clase que ele pertence, 
    x e y são as colunas do dataframe que se deseja comparar
    '''
    df_class_1 = df.loc[df['target'] == 1.0]
    df_class_2 = df.loc[df['target'] == 2.0]
    df_class_3 = df.loc[df['target'] == 3.0]
    
    plt.scatter(df_class_1[x], df_class_1[y], c='b')
    plt.scatter(df_class_2[x], df_class_2[y], c='r')
    plt.scatter(df_class_3[x], df_class_3[y], c='g')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
```


```python
piramide_geometrica = lambda alpha, ni , no: ceil(alpha * (( ni * no )**0.5))
```


```python
def creat_layers_set(num):
    layers = []
    aux_num = 0
    while num >= 0:
        if(num != 0 and aux_num != 0):
            layers.append((num, aux_num))
        elif(aux_num == 0):
            aux = "(" + str(num) + ",)"
            layers.append(eval(aux))
        num -= 1
        aux_num += 1
        
    return layers
```


```python
get_layers = lambda alpha, ni, no: creat_layers_set(piramide_geometrica(alpha, ni, no))
```

# 3. Fase de limpeza dos dados

Inicialmente nos iremos tranformar o dataset original que esta separado por tabs em um dataset separado por virgulas para que ele possa ser manipulado com mais facilidade


```python
df = cast_tab_to_comma_separated_values("dataset.txt")
```

Para facilitar a analize dos dados aqui nos iremos nomear cada coluna


```python
df.columns = ["area", "perimeter",
              "compactness", 
              "length_of_kernel", 
              "width_of_kernel", 
              "asymmetry_coefficient",
              "length_of_kernel_groove", 
              "target"]
```

Para que seja possivel fazer uma classificação e nao uma regressão aqui o atributo tarqet sera tranformado do tipo float para o tipo string


```python
df['target'] = df['target'].apply(lambda x: str(int(x)))
```

Visando evitar erros nos testes este dataset sera randomizado


```python
df = shuffle(df)
```

Agora temos o dataset pronto para analiza e treinamento dos modelos


```python
df.head()
```




<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>perimeter</th>
      <th>compactness</th>
      <th>length_of_kernel</th>
      <th>width_of_kernel</th>
      <th>asymmetry_coefficient</th>
      <th>length_of_kernel_groove</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>14.11</td>
      <td>14.26</td>
      <td>0.8722</td>
      <td>5.520</td>
      <td>3.168</td>
      <td>2.688</td>
      <td>5.219</td>
      <td>1</td>
    </tr>
    <tr>
      <th>151</th>
      <td>12.01</td>
      <td>13.52</td>
      <td>0.8249</td>
      <td>5.405</td>
      <td>2.776</td>
      <td>6.992</td>
      <td>5.270</td>
      <td>3</td>
    </tr>
    <tr>
      <th>99</th>
      <td>18.72</td>
      <td>16.34</td>
      <td>0.8810</td>
      <td>6.219</td>
      <td>3.684</td>
      <td>2.188</td>
      <td>6.097</td>
      <td>2</td>
    </tr>
    <tr>
      <th>47</th>
      <td>14.99</td>
      <td>14.56</td>
      <td>0.8883</td>
      <td>5.570</td>
      <td>3.377</td>
      <td>2.958</td>
      <td>5.175</td>
      <td>1</td>
    </tr>
    <tr>
      <th>101</th>
      <td>17.99</td>
      <td>15.86</td>
      <td>0.8992</td>
      <td>5.890</td>
      <td>3.694</td>
      <td>2.068</td>
      <td>5.837</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# 4. Analize dos dados

O numero de instancias deste dataset é mostrado abaixo


```python
len(df)
```




    210



Dentre as 210 instancias não ha valores faltando em nenhuma das colunas


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 210 entries, 21 to 137
    Data columns (total 8 columns):
    area                       210 non-null float64
    perimeter                  210 non-null float64
    compactness                210 non-null float64
    length_of_kernel           210 non-null float64
    width_of_kernel            210 non-null float64
    asymmetry_coefficient      210 non-null float64
    length_of_kernel_groove    210 non-null float64
    target                     210 non-null object
    dtypes: float64(7), object(1)
    memory usage: 14.8+ KB
    

Existem exatamente 70 exemplos de cada classe, desta forma este é um dataset balanciado


```python
df["target"].value_counts()
```




    1    70
    3    70
    2    70
    Name: target, dtype: int64



A matriz abaixo é muito util para mostrar corelações entre todos os elementos combinados dois a dois, atravez dela é possivel entender de uma forma ituitiva se as classes são separaveis em duas dimensões quando combinadas dessa forma.

- Por se tratar de um dataset com tres classes não há como separa-los utilizando apenas uma linha reta.
- No caso deste dataset é possível ver de forma intuitiva quais grupos são mais facilmente separaveis
- Os grupos que apresentam pontos mais juntos estão mais relacionados
- Os graficos mostram que o atributo **area** esta altamente relacionada as atributos **perimeter**, **length_of_kernel**, **width_of_kernel** e **length_of_kernel_groove**. Esta alta correlação provavelmente ocorre pelo fato de a area ser um atributo calculado atravez desses parametros
- O atributo **perimeter** esta tambem altamente correlacionado com os atributos **length_of_kernel** e **width_of_kernel** provavelmente pelo fato de ele ser calculado atravez desses atributos

### 4.1 pairplot 1


```python
sns.pairplot(df, hue="target")
```




    <seaborn.axisgrid.PairGrid at 0x239df580588>




![png](output_31_1.png)


Para ter uma noção mais precisa sobre a corelação entre os elementos combinados dois a dois aqui nos utilizamos uma tabela com o calculo das correlações


```python
corr = df.corr()
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>perimeter</th>
      <th>compactness</th>
      <th>length_of_kernel</th>
      <th>width_of_kernel</th>
      <th>asymmetry_coefficient</th>
      <th>length_of_kernel_groove</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>area</th>
      <td>1.000000</td>
      <td>0.994341</td>
      <td>0.608288</td>
      <td>0.949985</td>
      <td>0.970771</td>
      <td>-0.229572</td>
      <td>0.863693</td>
    </tr>
    <tr>
      <th>perimeter</th>
      <td>0.994341</td>
      <td>1.000000</td>
      <td>0.529244</td>
      <td>0.972422</td>
      <td>0.944829</td>
      <td>-0.217340</td>
      <td>0.890784</td>
    </tr>
    <tr>
      <th>compactness</th>
      <td>0.608288</td>
      <td>0.529244</td>
      <td>1.000000</td>
      <td>0.367915</td>
      <td>0.761635</td>
      <td>-0.331471</td>
      <td>0.226825</td>
    </tr>
    <tr>
      <th>length_of_kernel</th>
      <td>0.949985</td>
      <td>0.972422</td>
      <td>0.367915</td>
      <td>1.000000</td>
      <td>0.860415</td>
      <td>-0.171562</td>
      <td>0.932806</td>
    </tr>
    <tr>
      <th>width_of_kernel</th>
      <td>0.970771</td>
      <td>0.944829</td>
      <td>0.761635</td>
      <td>0.860415</td>
      <td>1.000000</td>
      <td>-0.258037</td>
      <td>0.749131</td>
    </tr>
    <tr>
      <th>asymmetry_coefficient</th>
      <td>-0.229572</td>
      <td>-0.217340</td>
      <td>-0.331471</td>
      <td>-0.171562</td>
      <td>-0.258037</td>
      <td>1.000000</td>
      <td>-0.011079</td>
    </tr>
    <tr>
      <th>length_of_kernel_groove</th>
      <td>0.863693</td>
      <td>0.890784</td>
      <td>0.226825</td>
      <td>0.932806</td>
      <td>0.749131</td>
      <td>-0.011079</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 4.2 pairplot 2

Consideranco a grande correlação entre o atributo **area** e outros atributos pode ser interessante separa-lo do resto para observar qual o efeito disso


```python
sns.pairplot(df[df.columns[1:]], hue="target")
```




    <seaborn.axisgrid.PairGrid at 0x239e2e20198>




![png](output_36_1.png)


### 4.3 pairplot 3

A partir dos graficos acima é possivel ver que **perimeter** tambem possui uma auta correlação com **length_of_kernel** e **width_of_kernel**, por iremos retirar o atributo **perimeter** e gerar o grafico novamente para poder observar os efeitos da mudança


```python
sns.pairplot(df[df.columns[2:]], hue="target")
```




    <seaborn.axisgrid.PairGrid at 0x239e2e20d68>




![png](output_39_1.png)


### 4.4 Hipóteses de modelos

Agora temos tres possiveis datasets

- O dataset original sem nenhuma alteração
- O dataset sem o atributo **area**
- O dataset sem os atributos **area** e **perimeter**

Assim partindo do principio de que informações muito corelacionadas serão mais difíceis de serem classificadas por uma rede neural nos testaremos essas tres configurações de dataset para verificar se a retirada desses parametros afeta possitivamente o treinamento dos modelos.

# 5. Testes de modelos

Aqui criamos um X para cada configuração proposta


```python
X_1 = df[df.columns[:-1]]
X_2 = X_1[X_1.columns[1:]]
X_3 = X_1[X_1.columns[2:]]
y = df[df.columns[-1:]]
```

A quantidade de neuronios foi definida atravez da regra da piramide geometrica, aqui implementada na função **get_layers** que foi definida no topico [2. funções uteis](#2.-funções-uteis), ela retorna uma lista com as combinações de tuplas que podem ser feitas de tal forma que o numero total de neuronios é igual ao resultado da regra da piramide


```python
output_size = 3
input_size_x_1 = X_1.shape[1]
input_size_x_2 = X_2.shape[1]
input_size_x_3 = X_3.shape[1]
```


```python
layers_1 = []
layers_2 = []
layers_3 = []

layers_1 += get_layers(0.5, input_size_x_1, output_size)
layers_1 += get_layers(2, input_size_x_1, output_size)
layers_1 += get_layers(3, input_size_x_1, output_size)

layers_2 += get_layers(0.5, input_size_x_2, output_size)
layers_2 += get_layers(2, input_size_x_2, output_size)
layers_2 += get_layers(3, input_size_x_2, output_size)

layers_3 += get_layers(0.5, input_size_x_3, output_size)
layers_3 += get_layers(2, input_size_x_3, output_size)
layers_3 += get_layers(3, input_size_x_3, output_size)
```

Aqui definiremos os parametros que serão combinados pelo **GridSearchCV**, criaremos tres conjuntos de parametros para testar as tres configurações de dataset propostas anteriormente


```python
activation_functions = ['identity', 'logistic', 'tanh', 'relu']
```

Utilizaremos o solver **lbfgs**, pois a [documentação](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) da classe **MLPClassifier** inidca que esse é o melhor solver para situações com datasets pequenos.


```python
parameters_1 = {'hidden_layer_sizes':layers_1, 
                'activation':activation_functions, 
                'solver':['lbfgs']}

parameters_2 = {'hidden_layer_sizes':layers_2, 
                'activation':activation_functions, 
                'solver':['lbfgs']}

parameters_3 = {'hidden_layer_sizes':layers_3, 
                'activation':activation_functions, 
                'solver':['lbfgs']}
```


```python
classifier_1 = GridSearchCV(MLPClassifier(), parameters_1, cv=3, scoring="accuracy")
classifier_2 = GridSearchCV(MLPClassifier(), parameters_2, cv=3, scoring="accuracy")
classifier_3 = GridSearchCV(MLPClassifier(), parameters_3, cv=3, scoring="accuracy")
```

### 5.1 Teste do modelo 1


```python
classifier_1.fit(X_1, y)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'hidden_layer_sizes': [(3,), (2, 1), (1, 2), (10,), (9, 1), (8, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (2, 8), (1, 9), (14,), (13, 1), (12, 2), (11, 3), (10, 4), (9, 5), (8, 6), (7, 7), (6, 8), (5, 9), (4, 10), (3, 11), (2, 12), (1, 13)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=0)



Acuracia:


```python
classifier_1.best_score_
```




    0.9666666666666667



### 5.2 Teste do modelo 2


```python
classifier_2.fit(X_2, y)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'hidden_layer_sizes': [(3,), (2, 1), (1, 2), (9,), (8, 1), (7, 2), (6, 3), (5, 4), (4, 5), (3, 6), (2, 7), (1, 8), (13,), (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6), (6, 7), (5, 8), (4, 9), (3, 10), (2, 11), (1, 12)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=0)



Acuracia:


```python
classifier_2.best_score_
```




    0.9571428571428572



### 5.3 Teste do modelo 3


```python
classifier_3.fit(X_3, y)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'hidden_layer_sizes': [(2,), (1, 1), (8,), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7), (12,), (11, 1), (10, 2), (9, 3), (8, 4), (7, 5), (6, 6), (5, 7), (4, 8), (3, 9), (2, 10), (1, 11)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=0)



Acuracia:


```python
classifier_3.best_score_
```




    0.9523809523809523



# 6. Conclusão


```python
quantity_of_models_1 = len(classifier_1.grid_scores_)
quantity_of_models_2 = len(classifier_2.grid_scores_)
quantity_of_models_3 = len(classifier_3.grid_scores_)
```

Os modelos propostos quando submetidos ao **GridSearch** geraram as seguintes quantidades de redes naurais


```python
print("classifier_1: " + str(quantity_of_models_1))
print("classifier_2: " + str(quantity_of_models_2))
print("classifier_3: " + str(quantity_of_models_3))
```

    classifier_1: 108
    classifier_2: 100
    classifier_3: 88
    

A quantidade total de modelos gerados foi:


```python
print("Total de modelos: " + str(quantity_of_models_1 + quantity_of_models_2 + quantity_of_models_3))
```

    Total de modelos: 296
    

Atravez da comparação entre as acuracias geradas utilizando as configurações de dataset propostas é possivel notar que a diferença entre eles é muito pequena, e por isso [proposição inicial](#4.4-Hipóteses-de-modelos) não se mostra interessante neste caso.

Tendo isso em vista iremos eleger o primeiro dataset, ou seja o dataset original com todos os dados, como melhor de todos, pois apesar de a area e o perimetro serem valores calculados a partir de outros valores as sementes não possuem formas muito regulares, assim nem toda as informações sobre as dimenções da semenete estão perfeitamente contidas em um unico parametro

Uma vez com o melhor dataset escolhido é necessario escolher qual a melhor conjunto de parametros, esta tarefa foi feita atravez da classe **GridSearchCV**, a melhor combinação escolhida levando em conta a acurácia e a acuracia foram:


```python
best_model = classifier_1.best_params_

print("Melhores Parametros: ")
print(" * Função de ativação: " + best_model["activation"])
print(" * Dimensões da camada oculta: " + str(best_model["hidden_layer_sizes"]))
print(" * Solver: " + best_model["solver"])

print()
print("Melhor acurácia: " + str(classifier_1.best_score_))
```

    Melhores Parametros: 
     * Função de ativação: identity
     * Dimensões da camada oculta: (11, 3)
     * Solver: lbfgs
    
    Melhor acurácia: 0.9666666666666667
    

Os valores acima podem variar de acordo com a execução
