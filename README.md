# MSDS - MLOps course

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)

## Development

To setup this repo locally, create a virtual environment (e.g. with [PyEnv](https://github.com/pyenv/pyenv)):

```bash
brew install pyenv
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
