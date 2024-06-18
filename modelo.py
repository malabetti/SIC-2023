import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def avaliar(name, chocolate_50, achocolatado, manteiga, tempo_de_cozimento):
    X = np.array([
        # chocolate em pó 50%, achocolatado, manteiga e tempo de cozimento #
        [1, 1, 2, 1, 8],
        [1, 1, 2, 1, 10],
        [1, 0, 3, 1, 8],
        [1, 0, 3, 1, 10],
        [1, 2, 1, 1, 8],
        [1, 2, 1, 1, 10],
        [1, 3, 0, 1, 8],
        [1, 3, 0, 1, 10],
        [1, 1.5, 1.5, 1, 8],
        [1, 1.5, 1.5, 1, 10],
        [1, 1, 2, 2, 8],
        [1, 1, 2, 2, 10],
        [1, 0, 3, 2, 8],
        [1, 0, 3, 2, 10],
        [1, 2, 1, 2, 8],
        [1, 2, 1, 2, 10],
        [1, 3, 0, 2, 8],
        [1, 3, 0, 2, 10],
        [1, 1.5, 1.5, 2, 8],
        [1, 1.5, 1.5, 2, 10], ])

    # média da avaliação de cada brigadeiro
    Y_sabor = np.array([
        [3.0833],
        [3.5],
        [3.5],
        [2.3333],
        [3.9167],
        [3.4167],
        [4.0833],
        [4.1818],
        [3.75],
        [3.8333],
        [3.4167],
        [3.5833],
        [3.25],
        [3.8182],
        [3.9167],
        [4.25],
        [3.8333],
        [3.75],
        [3.5833],
        [4.0833 ]	,	])

    Y_dulcor = np.array([
        [	2.833333333  ]	,
        [	2.75	]	,
        [	3.25]	,
        [	3  ]	,
        [	3.8333	]	,
        [	3.5833	]	,
        [	3.8333	]	,
        [	3.75	]	,
        [	3.5833	]	,
        [	3.5	]	,
        [	3.3333	]	,
        [	3.5	]	,
        [	3	]	,
        [ 3.75	]	,
        [	3.8333	]	,
        [	4.5	]	,
        [	3.9167	]	,
        [	4  ]	,
        [	3.5	]	,
        [	3.9167 	]	,	])

    Y_textura = np.array([
        [	2.5 ]	,
        [	3.416666667	]	,
        [	3.6667]	,
        [	1.6667 ]	,
        [	3.8333	]	,
        [	3.5	]	,
        [	3.8333	]	,
        [	3.6667	]	,
        [	3.75	]	,
        [	3.9167	]	,
        [	3.0833	]	,
        [	4.0833	]	,
        [	3.1667	]	,
        [ 4.1667	]	,
        [	3.3333	]	,
        [	3.8333	]	,
        [	3.4167	]	,
        [	4.0833  ]	,
        [	3.25	]	,
        [	4.5 	]	,	])

    Y_consistencia  = np.array([
        [	2.75 ]	,
        [	2.58333333	]	,
        [	3.41666667 ]	,
        [	 1.916666667 ]	,
        [	3.75	]	,
        [	3.5	]	,
        [	3.6666666667 ]	,
        [	3.8333333333	]	,
        [	3.5	]	,
        [	3.9166666667	]	,
        [	3.0833333333	]	,
        [	3.75	]	,
        [	2.9166666667	]	,
        [ 4.0833333333	]	,
        [	3.1666666667	]	,
        [	3.9166666667	]	,
        [	3.4166666667	]	,
        [	3.9166666667  ]	,
        [	3.0833333333	]	,
        [	4	 ]	,	])

    Y_aroma = np.array([
        [	3 ]	,
        [	3.5 ]	,
        [	3.08 ]	,
        [	2.417  ]	,
        [	3.333	]	,
        [	3	]	,
        [	3.667	]	,
        [	3.75	]	,
        [	3.5	]	,
        [	3	]	,
        [	3.583	]	,
        [	3.25	]	,
        [	2.909	]	,
        [ 3.583	]	,
        [	3.417	]	,
        [	3.083	]	,
        [	3	]	,
        [	3.667  ]	,
        [	3.667	]	,
        [	4.167 	]	,	])

    Y_cor = np.array([
        [	3 ]	,
        [	3.3	]	,
        [	3.3 ]	,
        [	2.1 ]	,
        [	4.1	]	,
        [	3.7	]	,
        [	4.3	]	,
        [	4.5	]	,
        [	3.8	]	,
        [	3.2	]	,
        [	3.4	]	,
        [	3.8	]	,
        [	3.6	]	,
        [ 3.6	]	,
        [	3.8	]	,
        [	4.3	]	,
        [	4.2	]	,
        [	4.6  ]	,
        [	3.7	]	,
        [	4.2 	]	,	])

    Y_nota_geral = np.array([
        [	3 ]	,
        [	3.166667	]	,
        [	3.333333 ]	,
        [	2.166667 ]	,
        [	4	]	,
        [	3.583333	]	,
        [	3.833333	]	,
        [	4	]	,
        [	3.416667	]	,
        [	3.583333	]	,
        [	3.166667	]	,
        [	3.666667	]	,
        [	2.833333	]	,
        [ 3.916667	]	,
        [	3.666667	]	,
        [	4	]	,
        [	3.666667	]	,
        [	4.25  ]	,
        [	3.166667	]	,
        [	4.25 	]	,	])

    data = np.concatenate((X, Y_sabor, Y_dulcor, Y_textura, Y_consistencia, Y_aroma, Y_cor, Y_nota_geral), axis=1)

    df = pd.DataFrame(data)
    df.columns = ['constante', 'chocolate em pó', 'achocolatado', 'manteiga', 'tempo', 'sabor', 'dulçor', 'textura', 'consitência', 'aroma', 'cor', 'nota geral']

    df.drop('constante', axis=1, inplace=True)

    X = df[['chocolate em pó', 'achocolatado', 'manteiga', 'tempo']]
    y = df[[name]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    # Criar um modelo de regressão linear
    model = LinearRegression()

    # Treinar o modelo usando os dados de treinamento
    model.fit(X_train, y_train)

    # Fazer previsões usando os dados de teste
    y_pred = model.predict(X_test)

    entrada = np.array([[chocolate_50, achocolatado, manteiga, tempo_de_cozimento]])


    previsao = model.predict(entrada)

    return previsao