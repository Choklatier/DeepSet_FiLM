import ROOT
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import plot_model
from hls4ml.model.profiling import numerical
import hls4ml
import numpy as np

import matplotlib.pyplot as plt

from AutoEncoder import build_qkeras_deepset_film, build_deepset_film


from DataProcessor import DataProcessor

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
print(trk_shift, trk_scale, event_shift, event_scale)
print("Transf:", 1/trk_scale, -trk_shift/trk_scale, 1/event_scale, -event_shift/event_scale)
# TODO : train multiple models using all folds
train_trk_array = folds[0][0]
train_event_array = folds[0][1]

# build model
# model = build_qkeras_deepset_film(
model = build_deepset_film(
    n_tracks_max=MAX_TRACKS,
    n_track_features=len(trk_columns),
    n_event_features=len(event_columns),
    latent_dim=8,
    # trk_shift=trk_shift,
    # trk_scale=trk_scale,
    # event_shift=event_shift,
    # event_scale=event_scale,
)


print(model.summary())
plot_model(
    model,
    to_file = "keras_model.pdf",
    show_shapes = True,
    show_dtype = True,
    expand_nested = True
)

# Forward pass test
B = 4
tracks = train_trk_array #np.random.randn(B,64, 5).astype(np.float32)
mask = np.ones((train_trk_array.shape[0],64,1)).astype(np.float32)
event = train_event_array #np.random.randn(B,3).astype(np.float32)
model.predict([tracks,mask,event])
# mu,logvar = model.predict([tracks,mask,event])
# print(mu,logvar)
# print(mu.shape, logvar.shape)
print("Forward pass test without crash!")

# Convert model to hls4ml
# Set options
config = hls4ml.utils.config_from_keras_model(
    model,
    granularity='name',
    default_precision='ap_fixed<16,6>',
)
# Optional: enforce precision
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 4
for layer in config['LayerName']:
    config['LayerName'][layer]['Trace'] = True # Debugging

config["LayerName"]["sum_over_tracks"]["Precision"] = {
    "result": "ap_fixed<24,10>",
    "weight": "ap_fixed<16,6>",
    "bias": "ap_fixed<1,1>",
    "accum": "ap_fixed<28,12>",
}

config["LayerName"]["dense_3"]["Precision"] = {
    "result": "ap_fixed<24,10>",
    "weight": "ap_fixed<16,6>",
    "bias": "ap_fixed<1,1>",
    "accum": "ap_fixed<28,12>",
}

config["LayerName"]["dense_3"]["Precision"] = {
    "result": "ap_fixed<24,10>",
    "weight": "ap_fixed<16,6>",
    "bias": "ap_fixed<1,1>",
    "accum": "ap_fixed<28,12>",
}

import pprint
pprint.pp(config["LayerName"]["sum_over_tracks"])
pprint.pp(config["LayerName"]["dense_3"])

# Convert model
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='deepset_film_hls4ml',
    part='xcu250-figd2104-2L-e'  # change to your FPGA
)

# Compile model 
hls_model.compile()
# print(config['LayerName'])
hls4ml.utils.plot_model(hls_model, show_shapes = True)
plots = numerical(model = model, hls_model = hls_model)
print(plots)
plt.show()

# run C simulation
y_hls = hls_model.predict([tracks, mask, event]) #.reshape(-1, 64, 16)
y_tf = model.predict([tracks, mask, event])
print(y_hls[0].shape, y_tf[0].shape)
# print(y_hls[0] - y_tf[0])
print(y_hls[0])
print("\n")
print(y_tf[0])
print(np.max(np.abs(y_hls[0] - y_tf[0])))


# for layer in model.layers:
#     print(layer.name, layer.output_shape)
# print(model.get_layer("sum_over_tracks").get_weights())

# _, hls_trace = hls_model.trace(
#     [tracks, mask, event]
# )

# keras_trace = hls4ml.model.profiling.get_ymodel_keras(
#     model,
#     [tracks, mask, event]
# )

# print("HLS TRACE:")
# print(hls_trace)
# print("KERAS TRACE:")
# print(keras_trace)

# Report
# hls_model.report()

# Synthesis
# hls_model.build(csim=True, synth=True, vsynth=True)