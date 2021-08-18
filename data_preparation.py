import pandas as pd
import logging
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import classification_report


logger = logging.getLogger(__name__)

def columns_to_dummies(df):
    """
    Essa função altera os labels das colunas de string para valores binários.
    O campo "Gender" é transformado com a função "get_dummies"

    Parameters
    ----------
    df : pandas.dataframe
        dataframe com todas as colunas utilizadas do projeto
    Returns
    -------
    df: pandas.dataframe
        dataframe com colunas com "yes" e "no" para 1 e 0, além do target "class" com valores transformados
        "Positive" para 1 e "Negative" para  0.
    """

    df['class']=df['class'].replace(['Positive'],1)
    df['class']=df['class'].replace(['Negative'],0)
    df=df.replace(['Yes'], 1)
    df=df.replace(['No'],0)
    df = pd.get_dummies(df, columns=['Gender'])
    return df

def transform_age(df):
    """
    Função que retorna a média de idade dos pacientes com casos positivos
    no dataset que está sendo avaliado.

    Parameters
    ----------
    df : pandas.dataframe
        dataframe com todas as colunas utilizadas do projeto
    Returns
    -------
    df: pandas.dataframe
        dataframe com a coluna "Age" na forma boolean
    """
    mean_age_positives = int(df.groupby(['class'])['Age'].mean()[1])
    logger.info(f'A média de idade dos pacientes positivos é de {mean_age_positives} anos')
    df['Age_mean'] = [1 if x >= int(df.groupby(['class'])['Age'].mean()[1]) else 0 for x in df.Age.values]
    return df

def featuring_select(df):
    """
    Seleciona variáveis importantes utilizando o método "KBest"

    Parameters
    ----------
    df : pandas.dataframe
        Dataframe pré processado

    Returns
    -------
    df: pandas.dataframe
        Dataframe com variáveis a serem utilizadas no modelo
    chi_features: list
        Lista com variáveis selecionadas pelo KBest
    """
    # Será considerado apenas o Gênero Feminino. Se 1 feminino, se 0 masculino
    df = df.drop(['Age', 'Gender_Male'], axis=1)
    X = df.drop('class', axis=1)
    y = df['class']
    chi_values = SelectKBest(chi2, k=11).fit(X, y)
    selected_features = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(chi_values.scores_)], axis=1)
    selected_features.columns = ["features", "values"]
    selected_features = selected_features.sort_values(by="values", ascending=False).reset_index(drop=False)
    logger.info(f'No teste com o "chi-quadrado", as variáveis selecionadas foram {selected_features["features"][0:-5].to_list()}')
    chi_features = selected_features["features"][0:-5].to_list()

    return df, chi_features

def train_model(X_train, X_test, y_train, y_test):
    """
    Parameters
    ----------
    X_train : list
        Lista contendo dados explicativos de treino
    X_test : list
        Lista contendo dados explicativos de treino
    y_train : list
        Lista contendo dados do target para treino
    y_test : list
        Lista contendo dados do target para teste

    Returns
    -------

    """
    params = {'n_estimators': [100, 300], 'max_depth': [2, 3, 4, 5], 'max_features': ['auto', 'sqrt', 'log2']}
    logger.info('Iniciando GridSearch')
    grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=42), params, verbose=1, cv=5)
    grid_search_cv.fit(X_train, y_train)
    logger.info('GridSearch e treino do modelo finalizado')
    rf_model = grid_search_cv.best_estimator_
    y_pred = rf_model.predict(X_test)
    target_names = ['negative', 'positive']
    logger.info(f'{classification_report(y_test, y_pred, target_names=target_names)}')
    feature_scores = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(
        ascending=False).to_frame()
    logger.info('Salvando modelo treinado')
    with open("./models/model.pkl","wb") as f:
        pickle.dump(rf_model,f)
    return logger.info(f'Variáveis mais importantes no modelo {feature_scores}')