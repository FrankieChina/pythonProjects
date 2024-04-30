import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

#0- Exploring Data
print("EXPLORING...")
df = pd.read_csv("DiabetesData2.csv")
print(df.head())
df.info()

#1- Standardize the features using StandardScaler.
def standardize_features():
    print("STANDARDIZING...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled

#2- Create 10 KMeans models on the scaled data with Ks (number of clusters) 
# in range(1,11). Note that 11 is excluded (Apply a loop to create 10 KMeans models).
def create_models(X_scaled):
    print("CREATE KMEANS MODEL...")
    k_values = range(1, 11)
    inertias = []

    for k in k_values:
        model = KMeans(n_clusters=k, n_init="auto")
        model.fit(X_scaled)
        inertias.append(model.inertia_)
    plot_inertias(k_values, inertias)

#3-Plot the inertias for 10 models with different Ks and see the results.
def plot_inertias(k_values, inertias):
    print("PLOT INERTIAS...")
    plt.plot(k_values, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.title('Inertias vs Clusters')
    plt.xticks(k_values)
    plt.show()

#4- Based on the resulted plot in the previous step, we now decide to group patients 
# in 6 clusters having patients with similar features to the same cluster. 
# For this create a single KMeans model with 6 clusters.
# Extract and store the cluster labels from the KMeans model in a variable and print 
# it out. What is the model inertia when K is 6?
def create_clusters(X_scaled):
    print("CREATE CLUSTERS...")
    model = KMeans(n_clusters=6, n_init='auto', random_state=42)
    model.fit(X_scaled)
    cluster_labels = model.labels_
    print("Cluster Labels: ", cluster_labels)
    print("Model Inertia when k is 6: ", model.inertia_)
    plot_clusters(X_scaled, cluster_labels)
    
def plot_clusters(X_scaled, labels):
    print("PLOT CLUSTERS...")
    xs = X_scaled[:,0]
    ys = X_scaled[:,2]
    plt.scatter(xs, ys, c=labels)
    plt.title('Clustered Data')
    plt.colorbar(label='Cluster')
    plt.show()

#BONUS- Create a plot of the variances of the PCA features (after scaling the data) and 
# interpret it.
def create_bonus(X_scaled):
    print("PCA VARIANCES...")
    scaler = StandardScaler()
    pca = PCA()
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(df)
    explained_variance = pca.explained_variance_
    features = range(pca.n_components_)
    plot_bonus(features, explained_variance)

def plot_bonus(features, explained_variance):
    print("PLOT PCA...")
    plt.bar(features, explained_variance)
    plt.xticks(features)
    plt.ylabel('variance')
    plt.xlabel('PCA feature')
    plt.title('PCA Feature Variances')
    plt.show()

def main():
    X_scaled = standardize_features()
    create_models(X_scaled)
    create_clusters(X_scaled)
    create_bonus(X_scaled)

if __name__ == "__main__":
    main()