import pandas as pd
import logging
from data_preparation import columns_to_dummies, transform_age, featuring_select, train_model
from sklearn.model_selection import train_test_split
import time

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='log.txt')
    logger = logging.getLogger(__name__)
    start = time.time()
    df = pd.read_csv('data\diabetes_data_upload.csv')
    logger.info(f'O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas')
    # transforma colunas para dummies
    logger.info('Transformando colunas para o formato de dummies')
    df = columns_to_dummies(df)
    # transformando a coluna de idade utilizando a média dos casos positivos
    df = transform_age(df)
    # retirando as 5 variáveis menos importante, através, do método "KBest"
    df, chi_features = featuring_select(df)
    # splitting dataset
    X = df[chi_features]
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    train_model(X_train, X_test, y_train, y_test)
    end = time.time()
    elapsed_time = end - start
    logger.info(f'Tempo de execução:{elapsed_time} segundos')
if __name__ == '__main__':
    main()
