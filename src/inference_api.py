import litserve as ls
import pickle
import numpy as np
from src.config_reader import Config

conf = Config()

class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu"):
        # load the model saved in above step during training
        with open("assets/model.pkl", "rb") as f:
            self.model = pickle.load(f)