# MSDS - MLOps course - Foodformer <img src="./images/foodformer_logo.jpeg" alt="foodformer_logo" width="20"/>

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nico-usf/foodformer)

## Development

To setup this repo locally, create a virtual environment (e.g. with [PyEnv](https://github.com/pyenv/pyenv)):

```bash
brew install pyenv
pyenv init
pyenv install -s 3.10.10
pyenv virtualenv 3.10.10 foodformer
pyenv activate foodformer
pyenv local foodformer
```

then install the dependencies and pre-commit hooks:

```bash
pip install -r requirements.txt
pre-commit install
```

## Testing the API

You can use API platforms like Postman or Insomnia, the command-line tool `curl`.

- for the healthcheck endpoint: `curl http://localhost:8000`
- for a post endpoint called `predict`: `curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H  "Content-Type: application/json" -d "{\"key\":\"value\",\"other_key\":\"other_value\"}"`
