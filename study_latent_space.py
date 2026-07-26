import ROOT
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

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

# Read settings from the config
with open("config.yaml","r") as config_file:
    config = yaml.safe_load(config_file)

# Read input information from yaml config
variables_to_define = config["Inputs"]["variables_to_define"]
trk_columns = config["Inputs"]["trk_columns"]
event_columns = config["Inputs"]["event_columns"]

MAX_TRACKS = config["Architecture"]["MAX_TRACKS"]
RHO_DIM = config["Architecture"]["RHO_DIM"]
LATENT_DIM = config["Architecture"]["LATENT_DIM"]

MAX_EVENTS = config["Training"]["MAX_EVENTS"]
K_FOLDS    = config["Training"]["K_FOLDS"]

# Prepare some input data
print("Reading data...")
root_file = ROOT.TFile.Open("Multijet_2010B.root")
tree = root_file.Get("analyzer/Events")
print("Processing data...")
DP = DataProcessor(
    tree,
    trk_columns,
    event_columns,
    variables_to_define = variables_to_define,
    max_events = MAX_EVENTS,
    max_tracks = MAX_TRACKS,
    )
print("Dividing data into folds...")
folds = DP.get_kfold_dataset(kfolds = K_FOLDS, cut = "1")

val_trk_array, val_event_array = folds[0]
train_trk_array, train_event_array = folds[1]

# Create the masks
val_valid_tracks = np.minimum(val_event_array[:, 2].astype(np.int32), MAX_TRACKS)
val_mask = (
    np.arange(MAX_TRACKS)[None, :] < val_valid_tracks[:, None]
).astype(np.float32)
val_mask = val_mask[..., None]

train_valid_tracks = np.minimum(train_event_array[:, 2].astype(np.int32), MAX_TRACKS)
train_mask = (
    np.arange(MAX_TRACKS)[None, :] < train_valid_tracks[:, None]
).astype(np.float32)
train_mask = train_mask[..., None]

# Convert data to float32
train_trk_array = train_trk_array.astype(np.float32, copy=False)
train_event_array = train_event_array.astype(np.float32, copy=False)
val_trk_array = val_trk_array.astype(np.float32, copy=False)
val_event_array = val_event_array.astype(np.float32, copy=False)
val_mask = val_mask.astype(np.float32, copy=False)

# Inject some LLPs
llp_maker = LLPMaker(val_trk_array, val_jets_array, trk_columns)


# Load model with ONNX to check it
onnx_model = onnx.load("deepset_film.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX checked!")

# Setup a session for inference
sess = ort.InferenceSession("deepset_film.onnx")
val_outputs = sess.run(
    None, {
        'tracks': val_trk_array,
        'mask': val_mask,
        'event': val_event_array,
        }
    )
train_outputs = sess.run(
    None, {
        'tracks': train_trk_array,
        'mask': train_mask,
        'event': train_event_array,
        }
    )

val_rho = val_outputs[0]
train_rho = train_outputs[0]
for i in range(RHO_DIM):
    for j in range(RHO_DIM):
        if i == j : continue

        val_ri = val_rho[::100,i]
        val_rj = val_rho[::100,j]

        train_ri = train_rho[::100,i]
        train_rj = train_rho[::100,j]

        plt.figure()
        plt.scatter(val_ri,val_rj, label = "validation")
        plt.scatter(train_ri,train_rj, label = "training")

        plt.title(f"$r_{i}$ - $r_{j}$")
        plt.xlabel(f"$r_{i}$")
        plt.ylabel(f"$r_{j}$")
        plt.legend()
        plt.savefig(f"plots/latent_space_study/rhodim{RHO_DIM}_r{i}_r{j}.pdf")