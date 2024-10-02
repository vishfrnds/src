from typing import Any
from cog import BasePredictor, Input, Path
from hub.hub import config_llama3_chat_7b


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        config_llama3_chat_7b.download_model()
        print('setup done')

    # Define the arguments and types the model takes as input
    def predict(self) -> Any:
        """Run a single prediction on the model"""
        return 'you are a cat'
