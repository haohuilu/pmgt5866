import numpy as np
import streamlit as st
from utils.functions import (
    get_model_tips,
    get_model_url,    train_model,
    plot_decision_boundary_and_metrics,
)

from utils.ui import (
    footer,
    generate_snippet,
    introduction,
    model_selector,
)

st.set_page_config(
    page_title="Playground for project analytics", layout="wide", page_icon="./images/flask.png"
)

import pandas as pd
from sklearn import model_selection

df = pd.read_csv("data.csv")

y = df["Class"]

def sidebar_controllers():
    model_type, model = model_selector()
    
    # Feature selection in the sidebar
    selected_features = st.sidebar.multiselect(
        "Select Features", 
        ["V-1", "V-2", "V-3", "V-4", "V-5", "V-6", "V-8"], 
        default=["V-1"]
    )

    footer()

    return model_type, model, selected_features

def body(model, model_type, selected_features):
    X = df[selected_features]  # Subset data based on the selected features
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=0.7, test_size=None, random_state=42
    )
    
    introduction()
    st.write(df.head(3))
    col1, col2 = st.columns((1, 1))

    with col1:
        plot_placeholder = st.empty()

    with col2:
        duration_placeholder = st.empty()
        model_url_placeholder = st.empty()
        code_header_placeholder = st.empty()
        snippet_placeholder = st.empty()
        tips_header_placeholder = st.empty()
        tips_placeholder = st.empty()

    model_url = get_model_url(model_type)

    (
        model,
        train_accuracy,
        train_f1,
        test_accuracy,
        test_f1,
        duration,
    ) = train_model(model, x_train, y_train, x_test, y_test)

    metrics = {
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
    }

    snippet = generate_snippet(model, model_type)
    model_tips = get_model_tips(model_type)
    
    fig = plot_decision_boundary_and_metrics(
        model, x_train, y_train, x_test, y_test, metrics
    )

    plot_placeholder.plotly_chart(fig, True)
    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    model_url_placeholder.markdown(model_url)
    tips_header_placeholder.header(f"**Tips on the {model_type} 💡 **")
    tips_placeholder.info(model_tips)

if __name__ == "__main__":
    model_type, model, selected_features = sidebar_controllers()
    body(model, model_type, selected_features)
