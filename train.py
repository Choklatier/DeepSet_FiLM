import tensorflow as tf
from tensorflow.keras import Model, layers
import hls4ml

from AutoEncoder import build_qkeras_deepset_film

# build model
model = build_qkeras_deepset_film(
    n_tracks_max = 64,
    n_track_features = 5,
    n_event_features = 3,
    latent_dim=8,
)

print(model.summary())


# Convert model to hls4ml
# Set options
config = hls4ml.utils.config_from_keras_model(
    model,
    granularity='name'
)
# Optional: enforce precision
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 1

# Convert model
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls4ml_prj',
    part='xcu250-figd2104-2L-e'  # change to your FPGA
)

# Compile model 
hls_model.compile()

# Report
hls_model.report()

# run C simulation
# y_hls = hls_model.predict([tracks, mask, event])

# Symthesis
hls_model.build(csim=True, synth=True, vsynth=True)