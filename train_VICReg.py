import ROOT
import tensorflow as tf
from tensorflow.keras import Model, layers
import hls4ml
import numpy as np

from DataProcessor import DataProcessor
from LLPMaker import LLPMaker
from AutoEncoder import build_qkeras_deepset_film, build_deepset_film

from VICReg_utility import rotate_phi_augmentation
from VICReg_utility import vicreg_loss

import yaml

with open("config.yaml","r") as config_file:
    config = yaml.safe_load(config_file)

# Read input information from yaml config
variables_to_define = config["Inputs"]["variables_to_define"]
trk_columns = config["Inputs"]["trk_columns"]
event_columns = config["Inputs"]["event_columns"]
jet_columns = config["Inputs"]["jet_columns"]

LOAD_ARRAYS = config["Inputs"]["LOAD_ARRAYS"]
SAVE_ARRAYS = config["Inputs"]["SAVE_ARRAYS"]
ARRAYS_FILEPATH = config["Inputs"]["ARRAYS_FILEPATH"]

MAX_TRACKS = config["Architecture"]["MAX_TRACKS"]
PHI_DIM = config["Architecture"]["PHI_DIM"]
RHO_DIM = config["Architecture"]["RHO_DIM"]
LATENT_DIM = config["Architecture"]["LATENT_DIM"]

MAX_EVENTS = config["Training"]["MAX_EVENTS"]
BATCH_SIZE = config["Training"]["BATCH_SIZE"]
N_EPOCHS   = config["Training"]["N_EPOCHS"]
K_FOLDS   = config["Training"]["K_FOLDS"]

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

# Read data
print("Reading data...")
root_file = ROOT.TFile.Open("Multijet_2010B.root")
tree = root_file.Get("analyzer/Events")
print("Processing data...")
DP = DataProcessor(
    tree,
    trk_columns,
    event_columns,
    variables_to_define = variables_to_define,
    jets_columns= jet_columns,
    max_events = MAX_EVENTS,
    max_tracks = MAX_TRACKS,
    filepath= ARRAYS_FILEPATH if LOAD_ARRAYS else None # Only indicate filepath if loading
    )

print("Dividing data into folds...")
folds = DP.get_kfold_dataset(kfolds = K_FOLDS, cut = "1")

# Save arrays if asked
if SAVE_ARRAYS:
    DP.save_arrays(ARRAYS_FILEPATH)

print("Computing linear transformation parameters...")
trk_shift, trk_scale, event_shift, event_scale = DP.get_lin_transform()
print(trk_shift, trk_scale, event_shift, event_scale)
print("Transf:", 1/trk_scale, -trk_shift/trk_scale, 1/event_scale, -event_shift/event_scale)

# Organise folds
# TODO : train multiple models using all folds
val_trk_array,val_event_array, _ = folds[0]
train_trk_array = []
train_event_array = []
for i in range(1,len(folds)):
    train_trk_array.append(folds[i][0])
    train_event_array.append(folds[i][1])
train_trk_array = np.concatenate(train_trk_array, axis = 0)
train_event_array = np.concatenate(train_event_array, axis = 0)

# Convert to float32 (float vs double conflicts)
train_trk_array = train_trk_array.astype(np.float32, copy=False)
train_event_array = train_event_array.astype(np.float32, copy=False)

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
    phi_dim= PHI_DIM,
    rho_dim= RHO_DIM,
    latent_dim=LATENT_DIM,
    trk_shift=trk_shift,
    trk_scale=trk_scale,
    event_shift=event_shift,
    event_scale=event_scale,
    vae_output=False, # No VAE output for VICReg method
)

print(model.summary())


# optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 1e-3) # Faster for M1/M2 Macs

for epoch in range(N_EPOCHS):

    for step, (trk_batch, event_batch) in enumerate(train_dataset):

        # Build mask
        valid_tracks = tf.cast(
            tf.minimum(
                tf.cast(event_batch[:, 2], tf.int32),
                MAX_TRACKS
            ),
            tf.int32,
        )

        mask_batch = tf.sequence_mask(
            valid_tracks,
            maxlen=MAX_TRACKS,
            dtype=tf.float32
        )

        mask_batch = tf.expand_dims(
            mask_batch,
            axis=-1
        )


        # Create second view
        trk_aug, event_aug = rotate_phi_augmentation(
            trk_batch,
            event_batch,
            trk_columns,
            event_columns
        )


        with tf.GradientTape() as tape:

            rho1 = model(
                [trk_batch,
                 mask_batch,
                 event_batch],
                training=True
            )

            rho2 = model(
                [trk_aug,
                 mask_batch,
                 event_aug],
                training=True
            )

            loss_value, sim_loss, var_loss, cov_loss = vicreg_loss(
                rho1,
                rho2
            )


        gradients = tape.gradient(
            loss_value,
            model.trainable_weights
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_weights
            )
        )

        # Compute variance of rho1 for logging
        std = tf.sqrt(
            tf.math.reduce_variance(rho1, axis=0)
        )
        
        if step % 10 == 0:
            print(
                f"epoch={epoch} "
                f"step={step} "
                f"Total loss={loss_value.numpy():.4f} "
                f"Inv ={sim_loss.numpy():.4f} ", 
                f"Var ={var_loss.numpy():.4f} ", 
                f"Cov ={cov_loss.numpy():.4f} ",
                f"std = {std.numpy()}",
            )


# Convert model and save to ONNX format
import tf2onnx

spec = (
    tf.TensorSpec(
        (None, MAX_TRACKS, len(trk_columns)),
        tf.float32,
        name="tracks",
    ),
    tf.TensorSpec(
        (None, MAX_TRACKS, 1),
        tf.float32,
        name="mask",
    ),
    tf.TensorSpec(
        (None, len(event_columns)),
        tf.float32,
        name="event",
    ),
)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,          # select operation set from ONNX
    output_path="deepset_film.onnx",
)