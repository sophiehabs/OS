# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 07:54:18 2020

@author: operard
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn import preprocessing
import numpy as np

# Using scikit-learn to perform K-Means clustering
from sklearn.cluster import KMeans
# librery to calculate error and visualization
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Process


def splitting(df, k):
    df_out = np.array_split(df, k)
    return df_out

# Function to iterate from k=2 until k=n with kmeans
def generate_kmeans(tabla_in,k,initialization,number_init, max_iterations,algoritmo):
    lista_modelos=[]
    lista_centroides=[]
    lista_silhouette=[]
    lista_inertia=[]
    for i in range(2,k+1):
        # for python version previously 3.9
        # kmeans_i=KMeans(n_clusters=i, init=initialization, n_init=number_init, max_iter=max_iterations, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm=algoritmo)
        # for python version 3.9 and scikit-learn 1.0.2
        kmeans_i=KMeans(n_clusters=i, init=initialization, n_init=number_init, max_iter=max_iterations, tol=0.0001,  verbose=0, random_state=None, copy_x=True, algorithm=algoritmo)
        kmedias_i=kmeans_i.fit(tabla_in)
        lista_modelos.append(kmedias_i)        
        lista_centroides.append(kmedias_i.cluster_centers_)
        lista_inertia.append(kmedias_i.inertia_)
        #lista_silhouette.append(silhouette_score(tabla_in,kmedias_i.labels_))
    return lista_modelos,lista_centroides,lista_silhouette,lista_inertia

def run(df):
    print (df.shape)

    # 2. Select Column for clustering
    colnames_pred = ['fullVisitorId','visitId','gadate','bounces','numhits','numpageviews',
                 'numevts','newVisits','screenviews','sessionQualityDim','timeOnScreen','timeOnSite',
                 'totalTransactionRevenue','transactionRevenue','transactions',
                    'UniqueScreenViews','visits','devicetype','mobileDeviceInfo' , 'mobileDeviceBranding','mobileDeviceModel',
                    'operatingSystem','browser','browserSize','screenResolution','Country','Region',
                    'City','City_ID']
    df_pred=df[colnames_pred].copy()

    #We transform the categorical variables columns into numerical values uing labelEncoder le()
    le = preprocessing.LabelEncoder()
    categorical_variables=['devicetype','mobileDeviceInfo' , 'mobileDeviceBranding','mobileDeviceModel','operatingSystem', 'browser','browserSize', 'screenResolution', 'Country','Region', 'City']

    for cat in categorical_variables:
        df_pred[cat] = le.fit_transform(df_pred[cat].astype(str))

    # Print Results Header    
    df_pred.head()

    # Clean Data
    df_pred = df_pred.fillna(df_pred.mean())
    #df_pred.drop('screenviews', axis=1, inplace=True)
    df_pred = df_pred.replace(np.nan,0)

    df_pred.head()


    # Code to scale all variables between 0 and 1
    scaler = MinMaxScaler()
    scaler.fit(df_pred[['numhits', 'numpageviews','totalTransactionRevenue','UniqueScreenViews','numevts']])
    df_pred_esc=scaler.transform(df_pred[['numhits', 'numpageviews','totalTransactionRevenue','UniqueScreenViews','numevts']])
    df_pred_escc = pd.DataFrame({'numhits': df_pred_esc[:, 0], 'numpageviews': df_pred_esc[:, 1], 'totalTransactionRevenue': df_pred_esc[:, 2], 'UniqueScreenViews': df_pred_esc[:, 3], 'numevts': df_pred_esc[:, 4]})

    # Recursive kmeans call to determine the better k
    init_time2 = datetime.datetime.now()
    num_clusters=10
    lista_modelos2,lista_centroides2,lista_silhouette2,lista_inertia2=generate_kmeans(tabla_in=df_pred_escc,k=num_clusters,initialization='k-means++',number_init=10, max_iterations=300,algoritmo='auto')
    end_time2 = datetime.datetime.now()
    dif_time2=end_time2-init_time2

    #print(dif_time2.seconds/60)#tiempo que tarda la ejecución en minutos

    temp=dif_time2.seconds
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    print('Execution Time: %d hour:%d min:%d sec' %(hours,minutes,seconds))

    # visualize the intracluster variance
    plt.scatter(range(2,num_clusters+1), lista_inertia2)

if __name__ == '_main_':
    # 1. Read CSV File to a Dataframe
    df = pd.read_csv('./sesion.zip',compression='zip',sep=',', low_memory=False);
    run(df)

    # SPlit the file into 5
    array_of_df = splitting(df, 5)

    # Array of processes to run
    processes = []

    # Start them
    for df in array_of_df:
        p = Process(target=run, args=(df,))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()