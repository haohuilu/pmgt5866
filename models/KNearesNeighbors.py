import streamlit as st
from sklearn.neighbors import KNeighborsClassifier


def knn_param_selector():

    n_neighbors = st.number_input("n_neighbors", 3)
    metric = st.selectbox(
        "metric", ("euclidean", "chebyshev", "mahalanobis")
    )
    weights = st.selectbox(
        "weights", ("distance", "uniform")
    )

    params = {"n_neighbors": n_neighbors, "metric": metric, "weights":weights}

    model = KNeighborsClassifier(**params)
    return model
