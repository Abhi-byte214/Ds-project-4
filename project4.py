import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data={'Annual Income':[27,56,76,98,34,76],'Spending Score':[88,54,76,43,98,56]}
df=pd.DataFrame(data)
X=df[['Annual Income','Spending Score']]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
kmeans=KMeans(n_clusters=3,random_state=42)
kmeans.fit(X_scaled)
df['cluster']=kmeans.labels_
cluster1=df[df['cluster']==0]
print(df)
print(cluster1)
print("Cluster 1 Data:")
cluster2=df[df['cluster']==1]
print(df)
print(cluster1)
print("Cluster 1 Data:")
cluster3=df [df ['cluster']==2]
print(df)