import streamlit as st
from sklearn.ensemble import RandomForestClassifier


def rf_param_selector():

    criterion = st.selectbox("criterion", ["gini"])
    n_estimators = st.number_input("n_estimators", 100)
    max_depth = st.number_input("max_depth", 10)
    max_features = st.selectbox("max_features", ["sqrt"])
    random_state = 0

    params = {
        "criterion": criterion,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
        "random_state": random_state,
    }

    model = RandomForestClassifier(**params)
    return model
