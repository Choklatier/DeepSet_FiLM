import ROOT
import tensorflow as tf
from tensorflow.keras import Model, layers
import hls4ml
import numpy as np
import matplotlib.pyplot as plt

from DataProcessor import DataProcessor
from LLPMaker import LLPMaker
from AutoEncoder import build_qkeras_deepset_film, build_deepset_film

from VICReg_utility import augment_tracks
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
ATTENTION_HIDDEN_DIM = config["Architecture"]["ATTENTION_HIDDEN_DIM"]
RHO_DIM = config["Architecture"]["RHO_DIM"]
LATENT_DIM = config["Architecture"]["LATENT_DIM"]

MAX_EVENTS = config["Training"]["MAX_EVENTS"]
BATCH_SIZE = config["Training"]["BATCH_SIZE"]
N_EPOCHS   = config["Training"]["N_EPOCHS"]
K_FOLDS   = config["Training"]["K_FOLDS"]
LAMBDA_INV = config["Training"]["LAMBDA_INV"]
LAMBDA_VAR = config["Training"]["LAMBDA_VAR"]
LAMBDA_COV = config["Training"]["LAMBDA_COV"]

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
    attention_hidden_dim=ATTENTION_HIDDEN_DIM,
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

total_losses = []
total_losses_std = []
inv_losses = []
inv_losses_std = []
var_losses = []
var_losses_std = []
cov_losses = []
cov_losses_std = []

for epoch in range(N_EPOCHS):
    total_loss_per_epoch = []
    inv_loss_per_epoch = []
    var_loss_per_epoch = []
    cov_loss_per_epoch = []
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

        # Create second two views
        trk_aug1, event_aug1, mask_aug1 = augment_tracks(
            trk_batch,
            event_batch,
            mask_batch,
            trk_columns,
            event_columns,
            boost_max= 0.1,
            track_mask_prob= None,
        )

        trk_aug2, event_aug2, mask_aug2 = augment_tracks(
            trk_batch,
            event_batch,
            mask_batch,
            trk_columns,
            event_columns,
            boost_max= 0.1,
            track_mask_prob= None,
        )


        with tf.GradientTape() as tape:

            rho1 = model(
                [trk_aug1,
                 mask_aug1,
                 event_aug1],
                training=True
            )

            rho2 = model(
                [trk_aug2,
                 mask_aug2,
                 event_aug2],
                training=True
            )

            loss_value, sim_loss, var_loss, cov_loss = vicreg_loss(
                rho1,
                rho2,
                lambda_inv=LAMBDA_INV,
                lambda_var=LAMBDA_VAR,
                lambda_cov=LAMBDA_COV,
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

        total_loss_per_epoch.append(loss_value.numpy())
        inv_loss_per_epoch.append(sim_loss.numpy())
        var_loss_per_epoch.append(var_loss.numpy())
        cov_loss_per_epoch.append(cov_loss.numpy())
        
        if step % 10 == 0:
            print(
                f"epoch={epoch} "
                f"step={step} "
                f"Total loss={loss_value.numpy():.4f} "
                f"Inv ={1000 * sim_loss.numpy():.4f} ", 
                f"Var ={var_loss.numpy():.4f} ", 
                f"Cov ={cov_loss.numpy():.4f} ",
                f"std = {std.numpy()}",
            )
    
    inv_loss_per_epoch = LAMBDA_INV * np.array(inv_loss_per_epoch)
    var_loss_per_epoch = LAMBDA_VAR * np.array(var_loss_per_epoch)
    cov_loss_per_epoch = LAMBDA_COV * np.array(cov_loss_per_epoch)

    total_losses.append(np.mean(total_loss_per_epoch))
    total_losses_std.append(np.std(total_loss_per_epoch))
    inv_losses.append(np.mean(inv_loss_per_epoch))
    inv_losses_std.append(np.std(inv_loss_per_epoch))
    var_losses.append(np.mean(var_loss_per_epoch))
    var_losses_std.append(np.std(var_loss_per_epoch))
    cov_losses.append(np.mean(cov_loss_per_epoch))
    cov_losses_std.append(np.std(cov_loss_per_epoch))
    print("Epoch summary:")
    print(
        f"epoch={epoch} "
        f"Mean Total Loss = {np.mean(total_loss_per_epoch):.4f} "
        f"Mean Inv Loss = {np.mean(inv_loss_per_epoch):.4f} "
        f"Mean Var Loss = {np.mean(var_loss_per_epoch):.4f} "
        f"Mean Cov Loss = {np.mean(cov_loss_per_epoch):.4f} "
        )
    print("")

plt.figure()
plt.errorbar(
    np.arange(0,len(total_losses_std),1),
    total_losses,
    yerr=total_losses_std,
    label = "Total loss",
    )
plt.errorbar(
    np.arange(0,len(total_losses_std),1),
    inv_losses,
    yerr=inv_losses_std,
    label = "Invariance loss",
    )
plt.errorbar(
    np.arange(0,len(total_losses_std),1),
    var_losses,
    yerr=var_losses_std,
    label = "Variance loss",
    )
plt.errorbar(
    np.arange(0,len(total_losses_std),1),
    cov_losses,
    yerr=cov_losses_std,
    label = "Covariance loss",
    )
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss.pdf")

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