import os
import random

from locust import HttpUser, between, events, task  # noqa: F401
from locust.env import Environment

# Run this in Docker with the following command:
# docker run -p 8089:8089 -v $PWD:/mnt/locust locustio/locust -f /mnt/locust/locustfile.py  # noqa: E501

IMAGES_FOLDER = "/Your/Local/Images/Folder"


@events.init.add_listener
def on_locust_init(environment: Environment, **kwargs: int) -> None:
    """Initialize the image filenames list in the environment to share it across
    users."""
    environment.filenames = os.listdir(IMAGES_FOLDER)


class QuickstartUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def call_root_endpoint(self) -> None:
        self.client.get("/")

    @task(3)  # 3 is the random task pick probability weight
    def call_predict(self) -> None:
        filename = self.get_random_image_filename()
        image_path = f"{IMAGES_FOLDER}/{filename}"
        ### EXERCISE: send a request to the /predict endpoint using ###
        ### Locust's self.client object ###

    def get_random_image_filename(self) -> str:
        return random.choice(self.environment.filenames)
