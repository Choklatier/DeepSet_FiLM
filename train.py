import tensorflow as tf
from tensorflow.keras import Model, layers
import hls4ml
import numpy as np

from AutoEncoder import build_qkeras_deepset_film, build_deepset_film

# build model
# model = build_qkeras_deepset_film(
model = build_deepset_film(
    n_tracks_max = 64,
    n_track_features = 5,
    n_event_features = 3,
    latent_dim=8,
)

print(model.summary())