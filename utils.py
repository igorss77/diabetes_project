import pandas as pd
import logging
logger = logging.getLogger(__name__)

def columns_to_dummies(df):
    """
    Essa função altera os labels das colunas de string para valores binários.
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