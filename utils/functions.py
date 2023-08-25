from pathlib import Path
import base64
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

from plotly.subplots import make_subplots
import plotly.graph_objs as go

from models.utils import model_infos, model_urls


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=True)
def generate_data(dataset, n_samples, train_noise, test_noise, n_classes):
    if dataset == "moons":
        x_train, y_train = make_moons(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_moons(n_samples=n_samples, noise=test_noise)
    elif dataset == "circles":
        x_train, y_train = make_circles(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_circles(n_samples=n_samples, noise=test_noise)
    elif dataset == "blobs":
        x_train, y_train = make_blobs(
            n_features=2,
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=train_noise * 47 + 0.57,
            random_state=42,
        )
        x_test, y_test = make_blobs(
            n_features=2,
            n_samples=n_samples // 2,
            centers=2,
            cluster_std=test_noise * 47 + 0.57,
            random_state=42,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def plot_decision_boundary_and_metrics(
    model, x_train, y_train, x_test, y_test, metrics
):
    d = x_train.shape[1]



    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Model Performance", None, None),
        row_heights=[0.3, 0.30],
    )




    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics["test_accuracy"],
            title={"text": f"Test Accuracy (%)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 100]}},
            number = {'valueformat':'.2f'},
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=metrics["test_f1"],
            title={"text": f"Test F1-score"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}},
            number = {'valueformat':'.4f'},
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=350,
    )

    return fig



def train_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 4)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="binary"), 4)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 4)*100
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="binary"), 4)

    return model, train_accuracy, train_f1, test_accuracy, test_f1, duration


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def get_model_tips(model_type):
    model_tips = model_infos[model_type]
    return model_tips


def get_model_url(model_type):
    model_url = model_urls[model_type]
    text = f"**Link to scikit-learn official documentation [here]({model_url}) 💻 **"
    return text


def add_polynomial_features(x_train, x_test, degree):
    for d in range(2, degree + 1):
        x_train = np.concatenate(
            (
                x_train,
                x_train[:, 0].reshape(-1, 1) ** d,
                x_train[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
        x_test = np.concatenate(
            (
                x_test,
                x_test[:, 0].reshape(-1, 1) ** d,
                x_test[:, 1].reshape(-1, 1) ** d,
            ),
            axis=1,
        )
    return x_train, x_test