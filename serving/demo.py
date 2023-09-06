import gradio as gr
import requests
import tensorflow as tf

inception_net = tf.keras.applications.MobileNetV2()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences


gr.Interface(
    fn=classify_image,
    inputs=gr.Image(
        shape=(224, 224), source="webcam", label="Upload Image or Capture from Webcam"
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicted Class"),
    examples=["/Users/nthiebaut/Downloads/image.jpg"],
    live=False,
).launch()
