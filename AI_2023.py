import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title('Tubes AI Lab 2024 | Explore different datasets')

# Select dataset name from selectbox
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

# Load dataset selected from sklearn
def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

# Data Description
X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))

# K as a parameter for KNN algorithm
params = dict()
K = st.sidebar.slider('K', 1, 15)
params['K'] = K
clf = KNeighborsClassifier(n_neighbors=params['K'])


#### CLASSIFICATION ####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = KNN')
st.write(f'Accuracy =', acc)

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)