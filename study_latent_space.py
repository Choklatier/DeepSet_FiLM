from DataProcessor import DataProcessor
from LLPMaker import LLPMaker

import onnx
import onnxruntime as ort

import yaml
"""
This script aims at studying the latent space of the DeepSet+FiLM VAE network.
The model, saved as an onnx format, is imported and tested with different inputs.
A main interest is to see the effect of injecting fake tracks that looks like LLPs.
"""


# Prepare some input data
with open("config.yaml","r") as config_file:
    config = yaml.safe_load(config_file)

# Read input information from yaml config
variables_to_define = config["Inputs"]["variables_to_define"]
trk_columns = config["Inputs"]["trk_columns"]
event_columns = config["Inputs"]["event_columns"]

MAX_TRACKS = config["Training"]["MAX_TRACKS"]



# Load model
onnx_model = onnx.load("deepset_film.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX checked!")

# Setup a session for inference

