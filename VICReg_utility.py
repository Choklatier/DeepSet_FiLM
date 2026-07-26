import tensorflow as tf
import numpy as np

# -------------------------------------------------
# trk_array & event augmentations
# -------------------------------------------------
def rotate_phi_augmentation(
        trk_batch, 
        event_batch,
        trk_columns,
        event_columns
        ):
    """
    Apply a global azimuthal rotation.
    Tracks and MET are rotated consistently.
    """

    delta_phi = tf.random.uniform(
        shape=(),
        minval=-np.pi,
        maxval=np.pi,
        dtype = np.float32
    )

    c = tf.cos(delta_phi)
    s = tf.sin(delta_phi)

    # Copy tensors
    trk_aug = tf.identity(trk_batch)
    event_aug = tf.identity(event_batch)

    # Rotate track px, py
    trk_px_idx = trk_columns.index("trk_px")
    trk_py_idx = trk_columns.index("trk_py")
    px = trk_batch[:, :, trk_px_idx]
    py = trk_batch[:, :, trk_py_idx]
    
    px_rot = c * px - s * py
    py_rot = s * px + c * py

    batch_size = tf.shape(px)[0]
    n_tracks = tf.shape(px)[1]
    event_idx, track_idx = tf.meshgrid(
        tf.range(batch_size),
        tf.range(n_tracks),
        indexing="ij",
    )

    track_indices_px = tf.stack([
        event_idx,
        track_idx,
        tf.fill(tf.shape(px), tf.constant(trk_px_idx, dtype=tf.int32)),
    ], axis=-1)
    track_indices_py = tf.stack([
        event_idx,
        track_idx,
        tf.fill(tf.shape(px), tf.constant(trk_py_idx, dtype=tf.int32)),
    ], axis=-1)

    trk_aug = tf.tensor_scatter_nd_update(
        trk_aug,
        tf.reshape(track_indices_px, [-1, 3]),
        tf.reshape(px_rot, [-1]),
    )
    trk_aug = tf.tensor_scatter_nd_update(
        trk_aug,
        tf.reshape(track_indices_py, [-1, 3]),
        tf.reshape(py_rot, [-1]),
    )

    # Rotate MET
    met_px_idx = event_columns.index("met_px")
    met_py_idx = event_columns.index("met_py")
    met_px = event_batch[:, met_px_idx]
    met_py = event_batch[:, met_py_idx]

    met_px_rot = c * met_px - s * met_py
    met_py_rot = s * met_px + c * met_py

    event_idx = tf.range(tf.shape(met_px)[0])
    event_indices_px = tf.stack([
        event_idx,
        tf.fill([tf.shape(met_px)[0]], tf.constant(met_px_idx, dtype=tf.int32)),
    ], axis=-1)
    event_indices_py = tf.stack([
        event_idx,
        tf.fill([tf.shape(met_px)[0]], tf.constant(met_py_idx, dtype=tf.int32)),
    ], axis=-1)

    event_aug = tf.tensor_scatter_nd_update(
        event_aug,
        event_indices_px,
        tf.reshape(met_px_rot, [-1]),
    )
    event_aug = tf.tensor_scatter_nd_update(
        event_aug,
        event_indices_py,
        tf.reshape(met_py_rot, [-1]),
    )

    return trk_aug, event_aug

# ----------------------------------------------------------------------
# VIGReg Losses
# ----------------------------------------------------------------------
def vicreg_variance_loss(x, gamma=1.0):

    std = tf.sqrt(
        tf.math.reduce_variance(x, axis=0)
        + 1e-4
    )

    return tf.reduce_mean(
        tf.nn.relu(gamma - std)
    )


def vicreg_covariance_loss(x):

    x = x - tf.reduce_mean(x, axis=0)

    n = tf.cast(tf.shape(x)[0], tf.float32)

    cov = tf.matmul(
        x,
        x,
        transpose_a=True
    ) / (n - 1)

    diag = tf.linalg.diag(
        tf.linalg.diag_part(cov)
    )

    off_diag = cov - diag

    return tf.reduce_sum(
        tf.square(off_diag)
    ) / tf.cast(tf.shape(cov)[0], tf.float32)


def vicreg_loss(
    rho1,
    rho2,
    lambda_inv=25.0,
    lambda_var=25.0,
    lambda_cov=1.0
):

    # Invariance
    sim_loss = tf.reduce_mean(
        tf.square(rho1 - rho2)
    )

    # Variance
    var_loss = (
        vicreg_variance_loss(rho1)
        +
        vicreg_variance_loss(rho2)
    )

    # Covariance
    cov_loss = (
        vicreg_covariance_loss(rho1)
        +
        vicreg_covariance_loss(rho2)
    )

    return (
        lambda_inv * sim_loss
        +
        lambda_var * var_loss
        +
        lambda_cov * cov_loss
    ), sim_loss, var_loss, cov_loss