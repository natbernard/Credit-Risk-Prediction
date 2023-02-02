import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('ML_challenge_dataset.xlsx')

def data_preprocessing(df):
    # renaming the frequency column
    df.rename(columns = {'customer seasonality (frequency of purchase)' : 'customer seasonality'},\
        inplace = True)

    # dropping records where no loan transactions occurred
    df = df.dropna(subset = ['amount requested', 'amount issued'])

    # dropping duplicates by the unique sale id column
    df.drop_duplicates(subset = 'unique sale id', keep = 'first', inplace = True)

    # setting null values to zero
    def set_to_zero(df, col1, col2):
        df.loc[df[col1].isnull(), col1] = 0
        df.loc[df[col1] == 0, col2] = 0
        return df

    # calling the function
    df = set_to_zero(df, 'paid amount', 'repayment days')

    # imputing the missing values on the 'repayment days' coluns
    df['repayment days'].fillna(df['repayment days'].mode()[0], inplace = True)

    # creating a function to correct misspelt values in customer seasonality and replace null values with the 'Daily' frequency
    def correct(df, col):
        df.loc[df[col] == 'Quaterly', col] = 'Quarterly'
        df.loc[df[col] == 'Monthy', col] = 'Monthly'
        df.loc[df[col].isnull(), col] = 'Daily'
        return df

    # calling the function
    correct(df, 'customer seasonality')

    # creating mappings for the customer seasonality column
    df['customer seasonality'] = df['customer seasonality'].map({
        'Daily': 0,
        'Weekly': 1,
        'Monthly': 2,
        'Quarterly':3
    })

    # creating the defaulting frequency column
    df['defaulted'] = df['paid amount'].apply(lambda x: 0 if x > 1 else 1)

    # creating the late payment column
    df['late payment'] = df['repayment days'].apply(lambda x: 0 if (x <= 14 and x > 1) else 1)

    # creating the defaulted amounts column
    df['amount defaulted'] = df.apply(lambda x: x['amount issued'] - x['paid amount'], axis = 1)

    # creating the repeat business column
    repeat = df[df['customer auto id'].duplicated()]['customer auto id']
    df['repeat business'] = df['customer auto id'].apply(lambda x: 1 if x in repeat.tolist() else 0)

    return df

def create_model(df):
    # dropping unnecessary colums
    df = df.drop(['unique sale id', 'good_bad_flag', 'customer auto id'], axis = 1)

    # scaling the data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    # dimensionality reduction
    pca = PCA(n_components = 0.9, random_state = 101)
    pca_df = pca.fit_transform(scaled_df)

    # fitting the model and making the prediction
    kmeans = KMeans(n_clusters = 2, random_state = 101)
    df['good_bad_flag'] = kmeans.fit_predict(pca_df)
    cluster_labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # inverting the clusters
    df['good_bad_flag'] = df['good_bad_flag'].apply(lambda x: 0 if x == 1 else 1)
    print(df.head())

    return df


def main():
    clean_df = data_preprocessing(df)
    create_model(clean_df)


if __name__ == "__main__":
    main()