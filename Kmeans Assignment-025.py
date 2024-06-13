#loading data
import pandas as pd
joan=pd.read_csv('OnlineRetail.csv',encoding="unicode_escape")
print(mutuku.head(10))
print(mutuku.shape)
print(mutuku.info())
print(mutuku.columns)
print(mutuku.describe())

#data cleaning
print(mutuku.isnull().sum())
mutuku_null=round(100*(mutuku.isnull().sum())/len(mutuku),2)#to give null values in percentage raised to 2dp
print(mutuku_null)
#mutuku=mutuku.drop(['StockCode'], axis=1)
# Changing the datatype of Customer Id as per Business understanding
mutuku['CustomerID']=mutuku['CustomerID'].astype(str)
print(mutuku)

#Data Preparation
#1)Recency(R)-number of days since last purchase
mutuku['Amount'] = mutuku['Quantity']*mutuku['UnitPrice']
print(mutuku.info())
print(mutuku.head())
mutuku.monitoring = mutuku.groupby('CustomerID')['Amount'].sum()
print(mutuku.monitoring)
print(mutuku.monitoring.head())
#mostsold product
mutuku.monitoring = mutuku.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print(mutuku.monitoring.head())
#region
mutuku.monitoring = mutuku.groupby('Country')['Quantity'].sum()
print(mutuku.monitoring.head())

#frequently sold
mutuku.monitoring = mutuku.groupby('Description')['InvoiceNo'].count()
print(mutuku.monitoring.head())
# Convert to datetime to proper datatype

mutuku['InvoiceDate'] = pd.to_datetime(mutuku['InvoiceDate'], format='%m/%d/%Y %H:%M')
print(mutuku)
# Compute the maximum date to know the last transaction date

max_date = max(mutuku['InvoiceDate'])
print(max_date)
# Compute the minimum date to know the last transaction date

min_date = min(mutuku['InvoiceDate'])
print(min_date)
mutuku['Diff'] = max_date - mutuku['InvoiceDate']
mutuku.head()

# The 'InvoiceDate' column is already in datetime format
date_difference = mutuku['InvoiceDate'].max() - mutuku['InvoiceDate'].min()

# Convert the difference to days
date_difference_days = date_difference.days
print("Difference between max and min dates:", date_difference_days, "days")

from datetime import timedelta
diff_time=max_date - timedelta(days=30)
print(diff_time)

dtt=mutuku[mutuku['InvoiceDate'] >diff_time]
total_amount = dtt['Amount'].sum()
print(total_amount)

total_amt_sales = dtt['Amount'].count()
print(total_amt_sales)


X = mutuku[['Quantity', 'UnitPrice']]  
y = mutuku['Amount']
#normalize data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
from sklearn import preprocessing
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

print("Size of training set:", X_train.shape)
print("Size of test set:", X_test.shape)




import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# Selecting features for clustering
X = mutuku[['Quantity', 'UnitPrice']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choosing the number of clusters (let's say 4 for demonstration)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Adding cluster labels to the dataset
mutuku['Cluster'] = kmeans.labels_

# Visualizing clusters
plt.figure(figsize=(10, 6))
for cluster in mutuku['Cluster'].unique():
    plt.scatter(mutuku[mutuku['Cluster'] == cluster]['Quantity'], mutuku[mutuku['Cluster'] == cluster]['UnitPrice'], label=f'Cluster {cluster}')
plt.xlabel('Quantity')
plt.ylabel('Unit Price')
plt.title('Clusters by Quantity and Unit Price')
plt.legend()
plt.show()




# Elbow Method for finding the optimal number of clusters

wcss = []  # Within-cluster sum of squares

# Selecting features for clustering
X = mutuku[['Quantity', 'UnitPrice']].dropna()

# Standardize the features
X_scaled = scaler.fit_transform(X)

# Trying different numbers of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()





























