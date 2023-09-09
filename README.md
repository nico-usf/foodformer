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

- for the healthcheck endpoint: `curl http://localhost:8080`
- for a post endpoint called `predict`:

```bash
curl -X 'POST' \
  'http://0.0.0.0:8080/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image.jpg;type=image/jpeg'
```

## Deployment to AWS Fargate

Fargate is a serverless deployment solution for Docker containers.

### Build and push the Docker image

In a terminal:

- Build the dockerfile: `docker build -t foodformer .`
- Fetch and store your AWS account id and AWS region: `export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text) && export AWS_REGION='us-east-2'`
- Authenticate with AWS ECR: `aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com` -> You should see the following message: "Login Succeeded"
- Create a repo in ECR: `aws ecr create-repository --repository-name foodformer --image-scanning-configuration scanOnPush=true --region $AWS_REGION`
- Tag and push you Docker image: `docker tag foodformer:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/foodformer` followed by `docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/foodformer` (this command will take a while, it's uploading the entire Docker image to ECR).

### Deployed API container with Fargate

Follow [this guide](https://app.tango.us/app/workflow/Creating-and-Deploying-an-ECS-Cluster-for-Foodformer-Application-7ede665f523044ec93f0239ad24f41a5).
