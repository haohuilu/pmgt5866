from sklearn import svm
import streamlit as st
from sklearn.svm import SVC


def svc_param_selector():
    C = st.number_input("C", 1)
    kernel = st.selectbox("kernel", ("rbf", "linear", "poly", "sigmoid"))
    params = {"C": C, "kernel": kernel, random_state = 1}
    model = SVC(**params)
    return model