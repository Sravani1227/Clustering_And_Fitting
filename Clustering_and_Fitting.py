# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sklearn.cluster as cluster
import matplotlib.pyplot as plt 
import sklearn.metrics as skmet
from scipy.optimize import curve_fit

# Min Max Scaler function squeezes the data to the range [0, 1]
def min_max_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

# This function plots the elbow plot of different clusters.
def elbow_plot(data):
    data = list(zip(data[:, 0], data[:, 1]))
    inertias = []
    for i in range(1,11):
        kmeans = cluster.KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

# Plotting k number of clusters.
def plot_kmeans(scaled_data, data, n):
    kmeans = cluster.KMeans(n_clusters = n)
    kmeans.fit(scaled_data)
    cen = kmeans.cluster_centers_
    # Printing the centers of the clusters.
    for i in range(n):
        print('The scaled coordinates of the center of cluster', i+1, 'are (', cen[i, 0], ',', cen[i, 1], ')')
    # Printing the silhouette score.
    print('The Silhouette score of the clusters = ', skmet.silhouette_score(scaled_data, kmeans.labels_))
    # Plotting the Original clusters.
    plt.scatter(data.iloc[:, 0].values, data.iloc[:, 1].values, c=kmeans.labels_)
    plt.xlabel('CO2 per capita consumption in metric tonnes')
    plt.ylabel('Electricity per capita consumption in kWhat')
    plt.title('Clusters found from K means algorithm')
    plt.show()

    # Plotting the scaled clusters.
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_)
    plt.xlabel('Scaled values of CO2 per capita consumption')
    plt.ylabel('Scaled values of Electricity per capita consumption')
    plt.title('Clusters found from K means algorithm')
    for i in range(n):
        plt.plot(cen[i, 0], cen[i, 1], "dk", markersize=10, c='b')
    plt.show()
    return kmeans

# Finding a sample from nth cluster.
def find_from_cluster(clusters, n):
    for i in range(len(clusters)):
        if (clusters[i] == n):
            return i

# This function determines the relation ship between x and y values of the courve y=f(x).
def objective(x, a, b):
    return a*x + b

# Curve fitting of CO2 data.
def fit_curve_co2(X, Y):
    popt, _ = curve_fit(objective, X, Y)
    a, b = popt
    plt.scatter(X, Y)
    plt.xlabel('Year')
    plt.ylabel('CO2 per capita consumption in metric tonnes per capita.')
    plt.title('CO2 per capita consumption in metric tonnes per capita in the country Albania')
    x_line = np.arange(min(X), max(X), 1)
    y_line = objective(x_line, a, b)
    plt.plot(x_line, y_line, '--', color='red')
    plt.show()
    # Predicting the future CO2 consumption from the curve.
    future_years = [2025, 2030, 2035, 2040, 2045, 2050]
    for i in range(len(future_years)):
        print('The predicted per capita CO2 consumption in Albaina in', future_years[i], 'is', objective(future_years[i], a, b))

# Comparing samples of different clusters.
def compare_countries(data, first_country, second_country):
    print(data.iloc[[first_country, second_country], :])

# Loading the data in csv file into a data frame object
co2_data = pd.read_csv('CO2_per_capita.csv')
electricity_data = pd.read_csv('Electric_power_consumption_per_capita.csv')

# Creating a data frame object by combining the above two data frames
data = pd.DataFrame({'Countries' : co2_data.iloc[:, 0].values,
                    'CO2 Emission' : co2_data.iloc[:, 58],
                    'Electricity Per Capita Consumption' : electricity_data.iloc[:, 58]})

# Deleting the rows with missing values
data = data.dropna()

# Scaling the data so as to eliminate the effect of units
scaled_data = min_max_scaler(data.iloc[:, 1:])

# Plotting the elbow plot to find the optimal number of clusters in K means algorithm.
elbow_plot(scaled_data)

# Plotting the K-means clusters
kmeans = plot_kmeans(scaled_data, data.iloc[:, 1:], 2)

# Comparing the members of differnt clusters.
country_from_first_cluster = find_from_cluster(kmeans.labels_, 0)
country_from_second_cluster = find_from_cluster(kmeans.labels_, 1)
compare_countries(data, country_from_first_cluster, country_from_second_cluster)

# Fitting a curve of CO2 data
fit_curve_co2(range(1990, 2020), co2_data.iloc[5, 34:64])