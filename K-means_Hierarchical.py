import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

#**************************************************** Load your dataset.**********************************************************
data = pd.read_csv('CC GENERAL.csv',encoding='UTF-8',delimiter= ',')
print(data)

# to know if we have missing values --> i found that Minimum_payments and Credit_limit has missing values
print((data.isnull().sum()/len(data)).sort_values(ascending=False))

#Missing values treatment
data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].median(),inplace=True)
data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(),inplace=True)
# to check if all attributes don't have a missing values
print('to check again', data.isnull().any())

# Drp the attributes CUST_ID
Data_clean= data.drop('CUST_ID',axis=1)


# Scaling data
sc_X = StandardScaler()
data_scaled = sc_X.fit_transform(Data_clean)
data_normalized = normalize(data_scaled)
data_normalized= pd.DataFrame(data_normalized)

#******************************************** Use hierarchical clustering to identify the inherent groupings within your data.***********

# reducing dimensionality
pca = PCA(n_components=2)
X_principal = pca.fit_transform(data_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(X_principal)

# Plot clusters
plt.scatter(X_principal['P1'], X_principal['P2'],
           c = AgglomerativeClustering(n_clusters = 5).fit_predict(X_principal), cmap =plt.cm.winter)
plt.show()

#************************************************** K-MEANS *****************************************************
plt.scatter(X_principal['P1'], X_principal['P2'],
           c = KMeans(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter)
plt.show()

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_principal)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()  #  So the idea of this algorithm is to choose the value of K at which the graph decrease the SSE the sum of squared distance
plt.scatter(X_principal['P1'], X_principal['P2'],
           c = KMeans(n_clusters = 9).fit_predict(X_principal), cmap =plt.cm.winter)
plt.show()
# the best one is k=9

# avec k=8 on a pu distinguer des classes homogenes dans le mm clusters et héterogenes entre les clusters.



