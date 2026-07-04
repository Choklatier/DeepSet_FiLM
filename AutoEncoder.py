import tensorflow as tf
from tensorflow.keras import Model, layers
from qkeras import QDense, QActivation, quantized_bits, quantized_relu

def build_qkeras_deepset_film(
    n_tracks_max,
    n_track_features,
    n_event_features,
    latent_dim=8
):

    # Inputs
    tracks_in = layers.Input(shape=(n_tracks_max, n_track_features), name="tracks")
    mask_in   = layers.Input(shape=(n_tracks_max, 1), name="mask")
    event_in  = layers.Input(shape=(n_event_features,), name="event")

    # ---- φ: per-track encoder ----
    x = tracks_in
    x = QDense(16, kernel_quantizer=quantized_bits(16,6,1), bias_quantizer=quantized_bits(16,6,1))(x)
    x = QActivation(quantized_relu(16,6))(x)
    x = QDense(16, kernel_quantizer=quantized_bits(16,6,1), bias_quantizer=quantized_bits(16,6,1))(x)
    x = QActivation(quantized_relu(16,6))(x)

    phi_dim = 16

    # ---- ψ: event → FiLM ----
    e = QDense(16, kernel_quantizer=quantized_bits(16,6,1), bias_quantizer=quantized_bits(16,6,1))(event_in)
    e = QActivation(quantized_relu(16,6))(e)

    gamma = QDense(phi_dim,
                   kernel_quantizer=quantized_bits(16,6,1),
                   bias_quantizer=quantized_bits(16,6,1),
                   name="gamma")(e)

    beta = QDense(phi_dim,
                  kernel_quantizer=quantized_bits(16,6,1),
                  bias_quantizer=quantized_bits(16,6,1),
                  name="beta")(e)

    gamma = layers.Reshape((1, phi_dim))(gamma)
    beta  = layers.Reshape((1, phi_dim))(beta)

    # ---- FiLM ----
    x = layers.Multiply()([x, gamma])
    x = layers.Add()([x, beta])

    # ---- Mask ----
    x = layers.Multiply()([x, mask_in])

    # ---- Sum pooling ----
    # x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="sum_pool")(x)
    # Swap to (batch, features, tracks)
    x = layers.Permute((2, 1), name="permute_tracks_features")(x)

    # Sum over tracks using a Dense layer with all-ones weights
    x = layers.Dense(
        1,
        use_bias=False,
        # kernel_initializer=tf.keras.initializers.Ones(),
        kernel_initializer=tf.keras.initializers.Constant(
            1.0/n_tracks_max
        ),
        trainable=False,
        name="sum_over_tracks"
    )(x)

    # # Now shape is (batch, features, 1)
    x = layers.Reshape((-1,), name="flatten_features")(x)

    # # ---- ρ: event-level ----
    x = QDense(16, kernel_quantizer=quantized_bits(16,6,1),
                     bias_quantizer=quantized_bits(16,6,1))(x)
    x = QActivation(quantized_relu(16,6))(x)

    # ---- Gaussian latent ----
    mu = QDense(latent_dim,
                kernel_quantizer=quantized_bits(16,6,1),
                bias_quantizer=quantized_bits(16,6,1),
                name="mu")(x)

    logvar = QDense(latent_dim,
                    kernel_quantizer=quantized_bits(16,6,1),
                    bias_quantizer=quantized_bits(16,6,1),
                    name="logvar")(x)

    # # ---- Gaussian NLL (approx anomaly score) ----
    # def gaussian_nll(args):
    #     mu, logvar = args
    #     return tf.reduce_sum(mu*mu * tf.exp(-logvar) + logvar, axis=1, keepdims=True)

    # score = layers.Lambda(gaussian_nll, name="score")([mu, logvar])

    return Model(inputs=[tracks_in, mask_in, event_in], outputs=[mu,logvar])

# Linear normalisation layer
class LinearNormalisation(layers.Layer):

    def __init__(self, scale, shift, **kwargs):
        super().__init__(trainable=False, **kwargs)

        self.scale = tf.constant(scale, dtype=tf.float32)
        self.shift = tf.constant(shift, dtype=tf.float32)
    
    def call(self, inputs):
        out = (inputs - self.shift) / self.scale
        return out


# Non-quantised version
def build_deepset_film(
    n_tracks_max,
    n_track_features,
    n_event_features,
    latent_dim=8,
    trk_shift=None,
    trk_scale=None,
    event_shift=None,
    event_scale=None,
):

    # Inputs
    tracks_in = layers.Input(
        shape=(n_tracks_max, n_track_features),
        name="tracks"
    )

    mask_in = layers.Input(
        shape=(n_tracks_max, 1),
        name="mask"
    )

    event_in = layers.Input(
        shape=(n_event_features,),
        name="event"
    )

    x = tracks_in
    e = event_in

    # ==========================================================
    # Normalisation layers (if given)
    # ==========================================================

    if trk_shift is not None and trk_scale is not None:
        x = LinearNormalisation(
            scale=trk_scale, 
            shift=trk_shift,
            name = "track_norm",
            )(tracks_in)

    if event_shift is not None and event_scale is not None:
        e = LinearNormalisation(
            scale=event_scale, 
            shift=event_shift,
            name = "event_norm"
            )(event_in)
    
    # ==========================================================
    # φ : per-track encoder
    # ==========================================================

    x = layers.Dense(16)(x)
    x = layers.ReLU()(x)

    x = layers.Dense(16)(x)
    x = layers.ReLU()(x)

    phi_dim = 16

    # ==========================================================
    # ψ : event -> FiLM parameters
    # ==========================================================

    e = layers.Dense(16)(e)
    e = layers.ReLU()(e)

    gamma = layers.Dense(
        phi_dim,
        name="gamma"
    )(e)

    beta = layers.Dense(
        phi_dim,
        name="beta"
    )(e)

    gamma = layers.Reshape((1, phi_dim))(gamma)
    beta = layers.Reshape((1, phi_dim))(beta)

    # ==========================================================
    # FiLM modulation
    # ==========================================================

    x = layers.Multiply()([x, gamma])
    x = layers.Add()([x, beta])

    # ==========================================================
    # Track mask
    # ==========================================================

    x = layers.Multiply()([x, mask_in])

    # ==========================================================
    # Deep Sets pooling
    # ==========================================================

    # Permute to (batch, features, tracks), to sum over tracks!
    x = layers.Permute(
        (2, 1),
        name="permute_tracks_features"
    )(x)

    # Dense layer with constant weights to sum over tracks
    x = layers.Dense(
        1,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(
            1.0 #/ n_tracks_max # TODO: Normalise with valid number of tracks instead?
        ),
        trainable=False,
        name="sum_over_tracks"
    )(x)

    x = layers.Reshape(
        (-1,),
        name="flatten_features"
    )(x)

    # ==========================================================
    # ρ : event-level network
    # ==========================================================

    x = layers.Dense(16)(x)
    x = layers.ReLU()(x)

    # ==========================================================
    # Gaussian latent parameters
    # ==========================================================

    mu = layers.Dense(
        latent_dim,
        name="mu"
    )(x)

    logvar = layers.Dense(
        latent_dim,
        name="logvar"
    )(x)

    return Model(
        inputs=[tracks_in, mask_in, event_in],
        outputs=[mu, logvar]
    )


def build_decoder(
    latent_dim=8,
    hidden_layers=[16, 16, 8],
    output_dim=3,
):

    decoder_in = layers.Input(
        shape=(latent_dim,),
        name="input"
    )

    x = decoder_in

    for layer_size in hidden_layers:
        x = layers.Dense(layer_size)(x)
        x = layers.ReLU()(x)

    x = layers.Dense(output_dim, name="decoder_output")(x)

    return Model(
        inputs=decoder_in,
        outputs=x,
    )


    