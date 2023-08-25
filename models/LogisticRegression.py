import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression


def lr_param_selector():

    solver = st.selectbox(
        "solver", options=["newton-cg"]
    )

    if solver in ["newton-cg", "lbfgs", "sag"]:
        penalties = ["l2", "none"]

    elif solver == "saga":
        penalties = ["l1", "l2", "none", "elasticnet"]

    elif solver == "liblinear":
        penalties = ["l2"]

    penalty = st.selectbox("penalty", options=penalties)
    #C = st.number_input("C", 100)
    #C = np.round(C, 3)
    random_state = 1
    #max_iter = st.number_input("max_iter", 100, 2000, step=50, value=100)

    params = {"solver": solver, "penalty": penalty, "C": 1, "random_state": random_state}

    model = LogisticRegression(**params)
    return model
