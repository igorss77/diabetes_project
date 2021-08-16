import pandas as pd
import numpy as np
import logging
from utils import columns_to_dummies, transform_age

handler = logging.FileHandler('log.txt', mode='w')


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='log.txt')
    logger = logging.getLogger(__name__)
    df = pd.read_csv('data\diabetes_data_upload.csv')
    logger.info(f'O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas')
    # transforma colunas para dummies
    logger.info('Transformando colunas para o formato de dummies')
    df = columns_to_dummies(df)
    # tranformando a coluna de idade utilizando a média dos casos positivos
    df = transform_age(df)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
