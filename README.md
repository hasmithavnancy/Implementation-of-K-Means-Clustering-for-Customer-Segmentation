# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Load the dataset (for example, Mall_Customers.csv).
3.Explore the dataset and select relevant features (like Annual Income and Spending Score). 4.Use the Elbow Method to find the optimal number of clusters (k).
4.Apply KMeans with the chosen k value to segment the customers.
5.Visualize the clusters using a scatter plot.
6.Interpret the results.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: HASMITHA V NANCY
RegisterNumber:  212224040111
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("/content/Mall_Customers.csv")   # path in Colab
data.head()

x = data.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids')

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()

```

## Output:
<img width="701" height="254" alt="image" src="https://github.com/user-attachments/assets/48dbe72e-e207-49f9-b259-703211448aed" />

<img width="597" height="455" alt="image" src="https://github.com/user-attachments/assets/9b4d51b8-7a97-497d-85ef-d54b2d1ea781" />
<img width="553" height="413" alt="image" src="https://github.com/user-attachments/assets/72a51555-d2cc-47e3-b22a-4550d1776b9e" />
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/e69fdfaf-fded-4718-a941-765b8a5632a8" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
