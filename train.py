import tensorflow as tf
from tensorflow.keras import Model, layers
import hls4ml
import numpy as np
import ROOT

from DataProcessor import DataProcessor
from AutoEncoder import build_qkeras_deepset_film, build_deepset_film, build_decoder

# Select data
variables_to_define = {
        "met_px" : "met_pt * cos(met_phi)",
        "met_py" : "met_pt * sin(met_phi)",
    }

trk_columns = [
    "trk_d0",
    "trk_pt",
    "trk_eta",
    "trk_phi",
]

event_columns = [
    "met_px",
    "met_py",
    "nTrack",
]

# Read data
root_file = ROOT.TFile.Open("Multijet_2010B.root")
tree = root_file.Get("analyzer/Events")
DP = DataProcessor(
    tree,
    trk_columns,
    event_columns,
    variables_to_define,
    max_events = 50000,
    )

folds = DP.get_kfold_dataset(kfolds = 5, cut = "1")

train_trk_array = folds[0][0]
train_event_array = folds[0][1]

# Split into batches
train_dataset = tf.Dataset.from_tensor_slices((train_trk_array, train_event_array))
train_dataset = train_dataset.batch(batch_size = 128)

# build model
# model = build_qkeras_deepset_film(
model = build_deepset_film(
    n_tracks_max = 64,
    n_track_features = 5,
    n_event_features = 3,
    latent_dim = 8,
)

print(model.summary())

# Get decoder
decoder = build_decoder(
    latent_dim = 8,
    hidden_layers = [16,8,4,2],
)

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

loss_huber = tf.keras.losses.Huber(from_logits = True)
loss_kl = tf.keras.losses.KLDivergence(from_logits = True)


for epoch in range(3):
    for step, (trk_train, event_train) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            
            # Go through the encoder TODO : Compute masks
            mu, log = model([trk_train, mask_train, event_train],training = True)

            # Go through the decoder TODO : change from mu to proper exp expression
            output = decoder(mu, training = True)

            # Compute loss TODO : add KL-divergence
            huber_value = loss_huber(mu, output)
            loss_value = huber_value

        # Get gradients and optimise network parameters
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply(gradients, model.trainable_weights)

        print(loss_value)


