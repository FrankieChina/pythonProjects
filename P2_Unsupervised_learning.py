import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("DiabetesData2.csv")
df.head()
df.info()

#1- Standardize the features using StandardScaler.
    # not sure what the target column is 
def standardize_features():
    target_column = "Age"
    X = df.drop(target_column, axis=1).values 
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled

#2- Create 10 KMeans models on the scaled data with Ks (number of clusters) 
# in range(1,11). Note that 11 is excluded (Apply a loop to create 10 KMeans models).
def create_models(X_train_scaled):
    inertias = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(X_train_scaled)
        inertias.append(kmeans.inertia_)
    plot_inertias(k_values, inertias)

#3-Plot the inertias for 10 models with different Ks and see the results.
def plot_inertias(k_values, inertias):
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('# of clusters')
    plt.ylabel('Inertia')
    plt.title('Inertias vs Clusters')
    plt.xticks(np.arange(1, 11, step=1))
    plt.show()

#4a- Based on the resulted plot in the previous step, we now decide to group patients 
# in 6 clusters having patients with similar features to the same cluster. 
# For this create a single KMeans model with 6 clusters.
# Extract and store the cluster labels from the KMeans model in a variable and print 
# it out. What is the model inertia when K is 6?
def create_clusters(X_train_scaled):
    kmeans_final = KMeans(n_clusters=6, n_init='auto', random_state=42)
    kmeans_final.fit(X_train_scaled)
    cluster_labels = kmeans_final.labels_
    print("Cluster Labels:", cluster_labels)
    print("Model Inertia when K=6:", kmeans_final.inertia_)

#BONUS- Create a plot of the variances of the PCA features (after scaling the data) and 
# interpret it.
def create_bonus(X_train_scaled):
    pca = PCA()
    pca.fit(X_train_scaled)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.title('PCA Feature Variances')
    plt.show()

def main():
    X_train_scaled = standardize_features()
    create_models(X_train_scaled)
    create_clusters(X_train_scaled)
    create_bonus(X_train_scaled)

if __name__ == "__main__":
    main()