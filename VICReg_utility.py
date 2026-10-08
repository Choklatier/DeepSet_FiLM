import tensorflow as tf
import numpy as np


def randomly_mask_track_instances(trk_batch, mask_batch, mask_probability=0.1):
    """Create a track/pair-dropout view while preserving feature values."""
    keep = tf.cast(
        tf.random.uniform(tf.shape(mask_batch), dtype=trk_batch.dtype)
        >= mask_probability,
        mask_batch.dtype,
    )
    return trk_batch, mask_batch * keep


def augment_track_pairs(
    pair_batch,
    context_batch,
    mask_batch,
    trk_columns,
    event_columns,
    jet_columns,
    phi_rotation_max=np.pi,
    eta_boost_max=0.1,
    pair_mask_probability=0.1,
):
    """Augment pT-ordered ``[track sum, signed track difference]`` pairs.

    A common azimuthal rotation rotates the PCA-coordinate sums and
    differences, leaves wrapped delta-phi invariant, and rotates the
    circular encoding of the phi sum by twice the angle. A common eta boost
    shifts the eta sum by twice the boost while leaving delta-eta unchanged.
    Event and jet context features receive the corresponding transformations.
    """
    n_features = len(trk_columns)
    expected_pair_features = 2 * n_features + ("trk_phi" in trk_columns)
    if pair_batch.shape[-1] != expected_pair_features:
        raise ValueError(
            f"Expected {expected_pair_features} pair features, "
            f"got {pair_batch.shape[-1]}"
        )
    if not 0 <= pair_mask_probability <= 1:
        raise ValueError("pair_mask_probability must be between 0 and 1")
    context_columns = list(event_columns) + list(jet_columns)
    if context_batch.shape[-1] != len(context_columns):
        raise ValueError("context feature count does not match event and jet columns")

    dtype = pair_batch.dtype
    valid_pairs = tf.squeeze(mask_batch > 0, axis=-1)
    pair_aug = tf.identity(pair_batch)
    context_aug = tf.identity(context_batch)
    phi_index = trk_columns.index("trk_phi") if "trk_phi" in trk_columns else None
    eta_index = trk_columns.index("trk_eta") if "trk_eta" in trk_columns else None

    def replace_feature(values, index, replacement):
        return tf.concat(
            (values[..., :index], replacement[..., tf.newaxis], values[..., index + 1:]),
            axis=-1,
        )

    delta_phi = tf.random.uniform(
        (),
        minval=-phi_rotation_max,
        maxval=phi_rotation_max,
        dtype=dtype,
    )
    cosine = tf.cos(delta_phi)
    sine = tf.sin(delta_phi)

    if ("trk_x_pca" in trk_columns) != ("trk_y_pca" in trk_columns):
        raise ValueError("trk_x_pca and trk_y_pca must both be configured for rotation")
    if "trk_x_pca" in trk_columns:
        x_index = trk_columns.index("trk_x_pca")
        y_index = trk_columns.index("trk_y_pca")
        for offset in (0, n_features):
            x = pair_aug[..., offset + x_index]
            y = pair_aug[..., offset + y_index]
            x_rotated = cosine * x - sine * y
            y_rotated = sine * x + cosine * y
            pair_aug = replace_feature(pair_aug, offset + x_index, x_rotated)
            pair_aug = replace_feature(pair_aug, offset + y_index, y_rotated)

    if phi_index is not None:
        phi_sum_cos_index = phi_index
        phi_sum_sin_index = 2 * n_features
        phi_sum_cos = pair_aug[..., phi_sum_cos_index]
        phi_sum_sin = pair_aug[..., phi_sum_sin_index]
        double_cosine = tf.cos(2 * delta_phi)
        double_sine = tf.sin(2 * delta_phi)
        pair_aug = replace_feature(
            pair_aug,
            phi_sum_cos_index,
            double_cosine * phi_sum_cos - double_sine * phi_sum_sin,
        )
        pair_aug = replace_feature(
            pair_aug,
            phi_sum_sin_index,
            double_sine * phi_sum_cos + double_cosine * phi_sum_sin,
        )

    context_columns = list(event_columns) + list(jet_columns)
    if ("met_px" in context_columns) != ("met_py" in context_columns):
        raise ValueError("met_px and met_py must both be configured for rotation")
    if "met_px" in context_columns:
        px_index = context_columns.index("met_px")
        py_index = context_columns.index("met_py")
        px = context_aug[..., px_index]
        py = context_aug[..., py_index]
        context_aug = replace_feature(
            context_aug, px_index, cosine * px - sine * py
        )
        context_aug = replace_feature(
            context_aug, py_index, sine * px + cosine * py
        )

    if "jet_phi" in jet_columns:
        jet_phi_index = len(event_columns) + jet_columns.index("jet_phi")
        jet_phi = context_aug[..., jet_phi_index] + delta_phi
        context_aug = replace_feature(
            context_aug,
            jet_phi_index,
            tf.atan2(tf.sin(jet_phi), tf.cos(jet_phi)),
        )

    boost = tf.random.uniform(
        (),
        minval=-eta_boost_max,
        maxval=eta_boost_max,
        dtype=dtype,
    )
    if eta_index is not None:
        eta_sum_index = eta_index
        eta_sum = pair_aug[..., eta_sum_index]
        eta_sum = eta_sum + tf.cast(valid_pairs, dtype) * (-2 * boost)
        pair_aug = replace_feature(pair_aug, eta_sum_index, eta_sum)
    if "jet_eta" in jet_columns:
        jet_eta_index = len(event_columns) + jet_columns.index("jet_eta")
        jet_eta = context_aug[..., jet_eta_index] - boost
        context_aug = replace_feature(context_aug, jet_eta_index, jet_eta)

    keep = tf.cast(
        tf.random.uniform(tf.shape(mask_batch), dtype=dtype)
        >= pair_mask_probability,
        mask_batch.dtype,
    )
    return pair_aug, context_aug, mask_batch * keep


# -------------------------------------------------
# trk_array & event augmentations
# -------------------------------------------------
def augment_tracks(
        trk_batch,
        event_batch,
        mask_batch,
        trk_columns,
        event_columns,
        boost_max = None,
        track_mask_prob = None,
        jet_columns = None,
        ):
    """
    Apply VICReg augmentations:

      1. Global azimuthal rotation
      2. Global longitudinal boost (implemented as eta shift)
      3. Random whole-track masking

    Rotate track transverse/PCA coordinates, MET, and any supplied jet
    azimuths together. Apply longitudinal boosts to track and jet eta.
    The event batch may include jet-context features after event features.

    Parameters
    ----------
    boost_max : float
        Maximum absolute boost rapidity y_b.
        y_b is sampled uniformly from [-boost_max, boost_max].

    track_mask_prob : float
        Probability of masking an otherwise-valid track.

    jet_columns : list, optional
        Column names for jet features appended to ``event_batch``.
    """

    # Copy tensors
    trk_aug = tf.identity(trk_batch)
    event_aug = tf.identity(event_batch)
    mask_aug = tf.identity(mask_batch)

    # ------------------------------------------------------------
    # 1. Global azimuthal rotation
    # ------------------------------------------------------------

    delta_phi = tf.random.uniform(
        shape=(),
        minval=-np.pi,
        maxval=np.pi,
        dtype = np.float32
    )

    c = tf.cos(delta_phi)
    s = tf.sin(delta_phi)

    def replace_feature(values, index, replacement):
        return tf.concat(
            (
                values[..., :index],
                replacement[..., tf.newaxis],
                values[..., index + 1:],
            ),
            axis=-1,
        )

    def rotate_pair(values, columns, x_name, y_name):
        has_x = x_name in columns
        has_y = y_name in columns
        if has_x != has_y:
            raise ValueError(f"{x_name} and {y_name} must both be configured")
        if not has_x:
            return values
        x_idx = columns.index(x_name)
        y_idx = columns.index(y_name)
        x = values[..., x_idx]
        y = values[..., y_idx]
        values = replace_feature(values, x_idx, c * x - s * y)
        return replace_feature(values, y_idx, s * x + c * y)

    trk_aug = rotate_pair(trk_aug, trk_columns, "trk_px", "trk_py")
    trk_aug = rotate_pair(trk_aug, trk_columns, "trk_x_pca", "trk_y_pca")

    if "trk_phi" in trk_columns:
        phi_idx = trk_columns.index("trk_phi")
        if trk_aug.shape[-1] == len(trk_columns) + 1:
            phi_sin = trk_aug[..., phi_idx]
            phi_cos = trk_aug[..., phi_idx + 1]
            trk_aug = replace_feature(
                trk_aug, phi_idx, s * phi_cos + c * phi_sin
            )
            trk_aug = replace_feature(
                trk_aug, phi_idx + 1, c * phi_cos - s * phi_sin
            )
        else:
            phi = trk_aug[..., phi_idx] + delta_phi
            trk_aug = replace_feature(
                trk_aug, phi_idx, tf.atan2(tf.sin(phi), tf.cos(phi))
            )

    # Rotate MET
    has_met_px = "met_px" in event_columns
    has_met_py = "met_py" in event_columns
    if has_met_px != has_met_py:
        raise ValueError("met_px and met_py must both be configured")
    if not has_met_px:
        raise ValueError("met_px and met_py are required for azimuthal rotation")
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

    jet_columns = jet_columns or []
    if "jet_phi" in jet_columns:
        jet_phi_idx = len(event_columns) + jet_columns.index("jet_phi")
        jet_phi = event_aug[..., jet_phi_idx] + delta_phi
        event_aug = replace_feature(
            event_aug,
            jet_phi_idx,
            tf.atan2(tf.sin(jet_phi), tf.cos(jet_phi)),
        )

    # ------------------------------------------------------------
    # 2. Longitudinal boost
    # ------------------------------------------------------------
    #
    # For relativistic tracks:
    #
    #     eta' ~= eta - y_b
    #
    # where y_b is the boost rapidity.
    #
    # ------------------------------------------------------------

    if boost_max is not None:
        eta_idx = trk_columns.index("trk_eta")

        # Sample one global boost for the entire batch
        y_b = tf.random.uniform(
            shape=(),
            minval=-boost_max,
            maxval=boost_max,
            dtype=trk_aug.dtype,
        )

        eta = trk_aug[:, :, eta_idx]
        eta_boost = eta - tf.cast(mask_batch[..., 0], trk_aug.dtype) * y_b

        # Replace the eta feature
        trk_aug = replace_feature(trk_aug, eta_idx, eta_boost)

        if "jet_eta" in jet_columns:
            jet_eta_idx = len(event_columns) + jet_columns.index("jet_eta")
            jet_eta = event_aug[..., jet_eta_idx] - y_b
            event_aug = replace_feature(event_aug, jet_eta_idx, jet_eta)

    # ------------------------------------------------------------
    # 3. Random whole-track masking
    # ------------------------------------------------------------
    if track_mask_prob is not None:
        # Randomly decide which track slots are kept
        # Shape: (batch_size, n_tracks)
        batch_size = tf.shape(trk_aug)[0]
        n_tracks = tf.shape(trk_aug)[1]
        random_keep = (
            tf.random.uniform(
                shape=(batch_size, n_tracks),
                dtype=trk_aug.dtype
            ) >= track_mask_prob
        )

        # Shape: (batch_size, n_tracks, 1)
        random_keep = random_keep[..., tf.newaxis]

        # Combine with the original validity mask
        mask_aug = mask_batch * tf.cast(random_keep, mask_batch.dtype)

    return trk_aug, event_aug, mask_aug

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
    lambda_inv,
    lambda_var,
    lambda_cov,
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


def chamfer_loss(targets, predictions, target_mask):
    """Compute symmetric squared Chamfer distance for padded sets."""
    target_mask = tf.cast(tf.squeeze(target_mask, axis=-1) > 0, tf.bool)
    target_count = tf.reduce_sum(tf.cast(target_mask, tf.int32), axis=1)
    prediction_mask = tf.sequence_mask(
        target_count,
        maxlen=tf.shape(predictions)[1],
    )

    pairwise_squared_distances = tf.reduce_sum(
        tf.square(predictions[:, :, tf.newaxis, :] - targets[:, tf.newaxis, :, :]),
        axis=-1,
    )
    large_distance = tf.cast(1e6, pairwise_squared_distances.dtype)

    prediction_to_target = tf.reduce_min(
        tf.where(
            target_mask[:, tf.newaxis, :],
            pairwise_squared_distances,
            large_distance,
        ),
        axis=2,
    )
    prediction_to_target = tf.where(
        prediction_mask,
        prediction_to_target,
        tf.zeros_like(prediction_to_target),
    )
    prediction_to_target = tf.reduce_sum(prediction_to_target, axis=1) / tf.maximum(
        tf.cast(target_count, pairwise_squared_distances.dtype),
        1.0,
    )

    target_to_prediction = tf.reduce_min(
        tf.where(
            prediction_mask[:, :, tf.newaxis],
            pairwise_squared_distances,
            large_distance,
        ),
        axis=1,
    )
    target_to_prediction = tf.where(
        target_mask,
        target_to_prediction,
        tf.zeros_like(target_to_prediction),
    )
    target_to_prediction = tf.reduce_sum(target_to_prediction, axis=1) / tf.maximum(
        tf.cast(target_count, pairwise_squared_distances.dtype),
        1.0,
    )

    return tf.reduce_mean(prediction_to_target + target_to_prediction)