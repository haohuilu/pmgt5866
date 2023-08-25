# Playground 🧪

Playground is a **streamlit** application that allows you to tinker with machine learning models from your browser.

So if you're a data science practitioner you should definitely try it out :wink:

_This app is inspired by the great Tensorflow [playground](https://playground.tensorflow.org/). The only difference here is that it addresses classical machine learning models_



## Reference

Right [here](https://playground-ml.herokuapp.com/)


## Run the app locally

Make sure you have pip installed with Python 3.

- install pipenv

```shell
pip install pipenv
```

- go inside the folder and install the dependencies

```shell
pipenv install
```

- run the app

```shell
streamlit run app.py
```

## Structure of the code

- `app.py` : The main script to start the app
- `utils/`
  - `ui.py`: UI functions to display the different components of the app
  - `functions.py`: for data processing, training the model and building the plotly graphs
- `models/`: where each model's hyper-parameter selector is defined
