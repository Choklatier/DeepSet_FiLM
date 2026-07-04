import ROOT
import tensorflow as tf
from tensorflow.keras import Model, layers
import hls4ml
import numpy as np

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

# Force eager execution for debugging
# This makes tensors concrete and allows .numpy() and simple prints inside the model.
# if hasattr(tf, "config") and tf.config.experimental_run_functions_eagerly:
#     tf.config.experimental_run_functions_eagerly(True)

# # For TensorFlow 2.x, eager execution is already enabled by default.
# # If you are on an older installation, this fallback is kept for compatibility.
# try:
#     tf.compat.v1.enable_eager_execution()
# except Exception:
#     pass

MAX_TRACKS = 64
MAX_EVENTS = 5000
BATCH_SIZE = 128

# Read data
print("Reading data...")
root_file = ROOT.TFile.Open("Multijet_2010B.root")
tree = root_file.Get("analyzer/Events")
print("Processing data...")
DP = DataProcessor(
    tree,
    trk_columns,
    event_columns,
    variables_to_define,
    max_events = MAX_EVENTS,
    max_tracks = MAX_TRACKS,
    )
print("Dividing data into folds...")
folds = DP.get_kfold_dataset(kfolds = 5, cut = "1")

print("Computing linear transformation parameters...")
trk_shift, trk_scale, event_shift, event_scale = DP.get_lin_transform()

# TODO : train multiple models using all folds
train_trk_array = folds[0][0]
train_event_array = folds[0][1]

print("Splitting data into batches...")
# Split into batches
train_dataset = tf.data.Dataset.from_tensor_slices((train_trk_array, train_event_array))
train_dataset = train_dataset.batch(batch_size = BATCH_SIZE)
print("Shape of train_trk_array: ", train_trk_array.shape)
print("Shape of train_event_array: ", train_event_array.shape)
print("Number of batches:",len(train_dataset))

print("Building the models...")
# build model
# model = build_qkeras_deepset_film(
model = build_deepset_film(
    n_tracks_max=MAX_TRACKS,
    n_track_features=len(trk_columns),
    n_event_features=len(event_columns),
    latent_dim=8,
    trk_shift=trk_shift,
    trk_scale=trk_scale,
    event_shift=event_shift,
    event_scale=event_scale,
)

print(model.summary())

# Get decoder
decoder = build_decoder(
    latent_dim=8,
    hidden_layers=[16, 16, 8],
    output_dim=len(event_columns),
)

# optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 1e-3) # Faster for M1/M2 Macs

recon_loss_fn = tf.keras.losses.Huber()
kl_weight = 1e-3

for epoch in range(3):
    for step, (trk_batch, event_batch) in enumerate(train_dataset):

        # Event level mask for valid tracks
        # [:,2] is the nTrack feature in the event input array
        # TODO : remove hardcoded event_batch indexing?
        valid_tracks = tf.cast(
            tf.minimum(tf.cast(event_batch[:, 2], tf.int32), MAX_TRACKS),
            tf.int32,
        )
        mask_batch = tf.sequence_mask(valid_tracks, maxlen=MAX_TRACKS, dtype=tf.float32)
        mask_batch = tf.expand_dims(mask_batch, axis=-1)

        with tf.GradientTape() as tape:
            # Get the latent parameters
            mu, logvar = model([trk_batch, mask_batch, event_batch], training=True)
    
            # Compute the gaussian latent variable z
            eps = tf.random.normal(shape=tf.shape(mu))
            z = mu + tf.exp(0.5 * logvar) * eps
            
            # Reconstruct the event-level features from the latent variable
            reconstruction = decoder(z, training=True)

            recon_loss = recon_loss_fn(event_batch, reconstruction)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            )
            
            loss_value = recon_loss + kl_weight * kl_loss

        trainable_vars = model.trainable_weights + decoder.trainable_weights
        gradients = tape.gradient(loss_value, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        if step % 10 == 0:
            print(f"epoch={epoch} step={step} loss={loss_value.numpy():.4f} recon={recon_loss.numpy():.4f} kl={kl_loss.numpy():.4f}")


