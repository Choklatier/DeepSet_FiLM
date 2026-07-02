import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import plot_model
from hls4ml.model.profiling import numerical
import hls4ml
import numpy as np

import matplotlib.pyplot as plt

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
plot_model(
    model,
    to_file = "keras_model.pdf",
    show_shapes = True,
    show_dtype = True,
    expand_nested = True
)

# Forward pass test
B = 4
tracks = np.random.randn(B,64, 5).astype(np.float32)
mask = np.ones((B,64,1)).astype(np.float32)
event = np.random.randn(B,3).astype(np.float32)
model.predict([tracks,mask,event])
# mu,logvar = model.predict([tracks,mask,event])
# print(mu,logvar)
# print(mu.shape, logvar.shape)
print("Forward pass test without crash!")

# Convert model to hls4ml
# Set options
config = hls4ml.utils.config_from_keras_model(
    model,
    granularity='name'
)
# Optional: enforce precision
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 4
for layer in config['LayerName']:
    config['LayerName'][layer]['Trace'] = True # Debugging

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