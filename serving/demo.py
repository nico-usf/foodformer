# mypy: ignore-errors
import json

import gradio as gr
import requests

host = "18.188.191.239"
api_token = ""


def call_api(filepath, host=host, api_token=api_token):
    url = f"http://{host}/predict"

    payload = {}
    files = [("file", ("image.jpg", open(filepath, "rb"), "image/jpeg"))]
    headers = {"Authorization": f"Bearer {api_token}"}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    return json.loads(response.text)["predictions"]


gr.Interface(
    fn=call_api,
    inputs=gr.Image(
        type="filepath", source="webcam", label="Upload Image or Capture from Webcam"
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicted Class"),
    examples=["examples/raclette.jpg", "examples/gyoza.jpeg", "examples/macarons.jpeg"],
    live=False,
).launch()
