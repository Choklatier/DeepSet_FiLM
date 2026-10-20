import numpy as np
import ROOT
from tqdm import tqdm

class DataProcessor:

    def __init__(
        self, 
        tree : ROOT.TTree, 
        trk_columns : list,
        event_columns : list,
        jets_columns : list = None,
        variables_to_define : dict = None,
        max_tracks : int = 64,
        max_events : int = None,
        filepath : str = None, # filepath to load data from
        ) -> None:

        self.trk_columns = trk_columns
        self.event_columns = event_columns
        self.jets_columns = jets_columns
        self.max_tracks = max_tracks
        self.filepath = filepath

        self.tree = tree
        print(f"Tree has {tree.GetEntries()} entries.")
        # Produce RDF to apply filters, define vars and get npy arrays
        self.rdf = ROOT.RDataFrame(self.tree)
        # Cap max events if asked
        if max_events is not None:
            self.rdf = self.rdf.Range(max_events)

        # Define variables if provided
        if variables_to_define is not None and self.filepath is None:
            for var in variables_to_define:
                expr = variables_to_define[var]
                self.rdf = self.rdf.Define(var,expr)
        
        # Store last computed arrays
        self.trk_array = None
        self.trk_pairs_array = None
        self.event_array = None
        self.jets_array = None
        self.jet_matched_trk_array = None
        self.jet_matched_trk_pairs_array = None
        self.jet_matched_track_counts = None

    # Function that reads the array for one track variable
    def tracks_to_array(self, array):
        output = None
        for event in tqdm(array):
            # cap maximum of tracks
            tracks = event[:self.max_tracks]
            # pad with zeros if we have less tracks
            if len(tracks) < self.max_tracks:
                tracks = np.pad(
                    tracks, 
                    (0, self.max_tracks - len(tracks)), 
                    constant_values=0 
                    )
            output = np.vstack([output, tracks]) if output is not None else tracks
        return output.T
    
    # Function that reads the array for one jet variable
    def jets_to_array(self, array, max_jets = 6):
        output = None
        for event in tqdm(array):
            jets = event[:max_jets] 
            # pad with zeros if we have less jets
            if len(jets) < max_jets:
                jets = np.pad(
                    jets, 
                    (0, max_jets - len(jets)), 
                    constant_values=0 
                    )
            output = np.vstack([output, jets]) if output is not None else jets
        return output.T


    def get_npy_arrays(self, cut = "1"):

        # If filepath indicated, simply read data from file
        if self.filepath is not None:
            return self.load_arrays(self.filepath)

        rdf_filtered = self.rdf.Filter(cut)
        trk_arrays = rdf_filtered.AsNumpy(self.trk_columns)
        event_arrays = rdf_filtered.AsNumpy(self.event_columns)
        jets_arrays = rdf_filtered.AsNumpy(self.jets_columns) if self.jets_columns is not None else None

        # Convert dict of arrays to stricly arrays
        print(f"Preparing {len(self.trk_columns)} track features:")
        tracks_var_arrays = [self.tracks_to_array(trk_arrays[column]) for column in self.trk_columns]
        trk_array_output = np.stack(tracks_var_arrays)

        # Event array are straight forwardly all the same shape
        print(f"Preparing the event features:")
        event_array_output = event_arrays[self.event_columns[0]]
        for column in self.event_columns[1:]:
            event_array_output = np.vstack([event_array_output,event_arrays[column]])

        # Jets array
        if self.jets_columns is not None:
            print(f"Preparing {len(self.jets_columns)} jet features:")
            jets_var_arrays = [self.jets_to_array(jets_arrays[column]) for column in self.jets_columns]
            jets_array_output = np.stack(jets_var_arrays).T
        else:
            jets_array_output = None
        
        # Store arrays as fields
        self.trk_array = trk_array_output.T
        self.event_array = event_array_output.T
        self.jets_array = jets_array_output
        
        return trk_array_output.T, event_array_output.T, jets_array_output

    def get_jet_matched_npy_arrays(self, cut="1", max_jets=6):
        """Return tracks grouped by their associated jet.

        The track result has shape ``(events, jets, tracks, features)``.
        Tracks with ``trk_jetIdx == -1`` and tracks assigned to jets beyond
        ``max_jets`` are omitted. ``trk_jetIdx`` is read internally and is
        not included among the returned track features.
        """
        if max_jets < 0:
            raise ValueError("max_jets must be non-negative")
        if not self.jets_columns:
            raise ValueError("jets_columns must be configured for jet matching")
        if self.filepath is not None:
            required = (
                "jet_matched_trk_array",
                "jet_matched_track_counts",
                "event_array",
                "jets_array",
            )
            with np.load(self.filepath) as data:
                missing = [name for name in required if name not in data]
                if missing:
                    raise ValueError(
                        "The configured array file does not contain jet-matched "
                        f"arrays; missing: {', '.join(missing)}"
                    )
                trk_array = data["jet_matched_trk_array"]
                event_array = data["event_array"]
                jets_array = data["jets_array"]
                track_counts = data["jet_matched_track_counts"]
            expected_shape = (
                event_array.shape[0],
                max_jets,
                self.max_tracks,
                len(self.trk_columns),
            )
            if trk_array.shape != expected_shape:
                raise ValueError(
                    "Cached jet-matched tracks have shape "
                    f"{trk_array.shape}, expected {expected_shape}"
                )
            if jets_array.shape != (
                event_array.shape[0],
                max_jets,
                len(self.jets_columns),
            ):
                raise ValueError("Cached jet features do not match the requested configuration")
            self.jet_matched_trk_array = trk_array
            self.jet_matched_track_counts = track_counts
            self.event_array = event_array
            self.jets_array = jets_array
            return trk_array, event_array, jets_array

        columns = list(dict.fromkeys(
            self.trk_columns
            + self.event_columns
            + self.jets_columns
            + ["trk_jetIdx"]
        ))
        rdf_filtered = self.rdf.Filter(cut)
        arrays = rdf_filtered.AsNumpy(columns)

        n_events = len(arrays[self.event_columns[0]])
        event_array = np.column_stack(
            [arrays[column] for column in self.event_columns]
        )
        jets_array = np.zeros(
            (n_events, max_jets, len(self.jets_columns)),
            dtype=float,
        )
        trk_array = np.zeros(
            (n_events, max_jets, self.max_tracks, len(self.trk_columns)),
            dtype=float,
        )
        track_counts = np.zeros((n_events, max_jets), dtype=np.int64)

        for event_index in range(n_events):
            jet_features = [
                np.asarray(arrays[column][event_index])
                for column in self.jets_columns
            ]
            n_event_jets = len(jet_features[0])
            if any(len(features) != n_event_jets for features in jet_features):
                raise ValueError("Configured jet feature branches have mismatched lengths")
            n_stored_jets = min(n_event_jets, max_jets)
            for feature_index, features in enumerate(jet_features):
                jets_array[event_index, :n_stored_jets, feature_index] = (
                    features[:n_stored_jets]
                )

            jet_indices = np.asarray(arrays["trk_jetIdx"][event_index])
            track_features = [
                np.asarray(arrays[column][event_index])
                for column in self.trk_columns
            ]
            if any(len(features) != len(jet_indices) for features in track_features):
                raise ValueError(
                    "Configured track feature branches and trk_jetIdx have "
                    "mismatched lengths"
                )

            for track_index, jet_index in enumerate(jet_indices):
                jet_index = int(jet_index)
                if jet_index < 0 or jet_index >= n_stored_jets:
                    continue
                track_slot = track_counts[event_index, jet_index]
                if track_slot >= self.max_tracks:
                    continue
                trk_array[event_index, jet_index, track_slot] = [
                    features[track_index] for features in track_features
                ]
                track_counts[event_index, jet_index] += 1

        self.jet_matched_trk_array = trk_array
        self.jet_matched_track_counts = track_counts
        self.event_array = event_array
        self.jets_array = jets_array
        return trk_array, event_array, jets_array

    def save_jet_matched_arrays(self, filepath):
        """Save arrays produced by ``get_jet_matched_npy_arrays`` for reuse."""
        if (
            self.jet_matched_trk_array is None
            or self.jet_matched_track_counts is None
            or self.event_array is None
            or self.jets_array is None
        ):
            raise ValueError(
                "Jet-matched arrays have not been computed; call "
                "get_jet_matched_npy_arrays() first"
            )
        np.savez_compressed(
            filepath,
            jet_matched_trk_array=self.jet_matched_trk_array,
            jet_matched_track_counts=self.jet_matched_track_counts,
            event_array=self.event_array,
            jets_array=self.jets_array,
        )

    def get_track_pairs(
        self,
        radius,
        max_pairs,
        cut="1",
        position_columns=("trk_x_pca", "trk_y_pca", "trk_z0"),
    ):
        """Build permutation-invariant feature vectors for nearby track pairs.

        A pair is selected when the Euclidean distance between its position
        coordinates is at most ``radius``. Each selected pair is represented
        by feature sums followed by signed
        differences. Pairs are ordered by descending track pT, with a
        deterministic feature-based tie-break. ``trk_phi`` uses a wrapped
        difference and circular sine/cosine sum encoding.
        """
        if radius < 0:
            raise ValueError("radius must be non-negative")
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if len(position_columns) != 3:
            raise ValueError("position_columns must contain three columns")
        n_pair_features = 2 * len(self.trk_columns) + (
            "trk_phi" in self.trk_columns
        )

        trk_array, event_array, jets_array = self.get_npy_arrays(cut)
        position_indices = []
        for column in position_columns:
            if column not in self.trk_columns:
                raise ValueError(f"Track position column '{column}' is not configured")
            position_indices.append(self.trk_columns.index(column))

        n_events, _, n_features = trk_array.shape
        trk_pairs_array = np.zeros(
            (n_events, max_pairs, n_pair_features),
            dtype=trk_array.dtype,
        )

        for event_index, tracks in enumerate(trk_array):
            valid_tracks = np.any(tracks != 0, axis=1)
            valid_indices = np.flatnonzero(valid_tracks)
            if len(valid_indices) < 2 or max_pairs == 0:
                continue

            valid_tracks = tracks[valid_indices]
            positions = valid_tracks[:, position_indices]
            pair_indices = np.triu_indices(len(valid_tracks), k=1)
            pair_distances = np.linalg.norm(
                positions[pair_indices[0]] - positions[pair_indices[1]],
                axis=1,
            )
            selected = np.flatnonzero(pair_distances <= radius)
            if len(selected) == 0:
                continue

            pair_features = self._build_pair_features(
                valid_tracks[pair_indices[0][selected]],
                valid_tracks[pair_indices[1][selected]],
            )
            ordering = np.lexsort(pair_features.T[::-1], axis=0)
            ordering = ordering[np.argsort(pair_distances[selected][ordering], kind="stable")]
            pair_features = pair_features[ordering[:max_pairs]]
            trk_pairs_array[event_index, :len(pair_features)] = pair_features

        self.trk_pairs_array = trk_pairs_array
        return trk_pairs_array, event_array, jets_array

    def _build_pair_features(self, tracks_1, tracks_2):
        """Build pT-canonical pair features with circular encoding for phi."""
        if "trk_pt" not in self.trk_columns:
            raise ValueError("trk_pt must be configured to build canonical track pairs")

        pt_index = self.trk_columns.index("trk_pt")
        swap = tracks_1[:, pt_index] < tracks_2[:, pt_index]
        tied_pt = tracks_1[:, pt_index] == tracks_2[:, pt_index]
        undecided = tied_pt.copy()
        phi_index = (
            self.trk_columns.index("trk_phi")
            if "trk_phi" in self.trk_columns
            else None
        )
        coordinate_indices = {
            self.trk_columns.index(column)
            for column in ("trk_x_pca", "trk_y_pca")
            if column in self.trk_columns
        }
        for feature_index in range(tracks_1.shape[1]):
            if (
                feature_index == pt_index
                or feature_index == phi_index
                or (phi_index is not None and feature_index in coordinate_indices)
            ):
                continue
            feature_tie_swap = undecided & (
                tracks_1[:, feature_index] < tracks_2[:, feature_index]
            )
            swap |= feature_tie_swap
            undecided &= tracks_1[:, feature_index] == tracks_2[:, feature_index]

        if phi_index is not None:
            phi_difference = np.arctan2(
                np.sin(tracks_1[:, phi_index] - tracks_2[:, phi_index]),
                np.cos(tracks_1[:, phi_index] - tracks_2[:, phi_index]),
            )
            swap |= undecided & (phi_difference < 0)

        first = np.where(swap[:, np.newaxis], tracks_2, tracks_1)
        second = np.where(swap[:, np.newaxis], tracks_1, tracks_2)
        sums = first + second
        differences = first - second
        if "trk_phi" not in self.trk_columns:
            return np.concatenate((sums, differences), axis=1)

        phi_1 = first[:, phi_index]
        phi_2 = second[:, phi_index]
        wrapped_difference = np.arctan2(
            np.sin(phi_1 - phi_2),
            np.cos(phi_1 - phi_2),
        )
        differences[:, phi_index] = wrapped_difference
        phi_sum = phi_1 + phi_2
        sums[:, phi_index] = np.cos(phi_sum)
        return np.concatenate(
            (sums, differences, np.sin(phi_sum)[:, np.newaxis]),
            axis=1,
        )

    def get_jet_matched_track_pairs(
        self,
        radius,
        max_pairs,
        cut="1",
        max_jets=6,
        position_columns=("trk_x_pca", "trk_y_pca", "trk_z0"),
    ):
        """Build nearby track pairs independently within each jet.

        The pair result has shape ``(events, jets, pairs, pair_features)``.
        It contains feature sums followed by signed differences. Pairs are
        ordered by descending track pT, with a deterministic tie-break. When
        ``trk_phi`` is configured, its difference is wrapped to the circle,
        and its sum is encoded as cosine and sine components.
        """
        if radius < 0:
            raise ValueError("radius must be non-negative")
        if max_pairs < 0:
            raise ValueError("max_pairs must be non-negative")
        if len(position_columns) != 3:
            raise ValueError("position_columns must contain three columns")

        position_indices = []
        for column in position_columns:
            if column not in self.trk_columns:
                raise ValueError(f"Track position column '{column}' is not configured")
            position_indices.append(self.trk_columns.index(column))

        trk_array, event_array, jets_array = self.get_jet_matched_npy_arrays(
            cut=cut,
            max_jets=max_jets,
        )
        n_events, n_jets, _, n_features = trk_array.shape
        n_pair_features = 2 * n_features + ("trk_phi" in self.trk_columns)
        trk_pairs_array = np.zeros(
            (n_events, n_jets, max_pairs, n_pair_features),
            dtype=trk_array.dtype,
        )
        for event_index in range(n_events):
            for jet_index in range(n_jets):
                n_tracks = self.jet_matched_track_counts[event_index, jet_index]
                tracks = trk_array[event_index, jet_index, :n_tracks]
                if n_tracks < 2 or max_pairs == 0:
                    continue

                positions = tracks[:, position_indices]
                pair_indices = np.triu_indices(n_tracks, k=1)
                pair_distances = np.linalg.norm(
                    positions[pair_indices[0]] - positions[pair_indices[1]],
                    axis=1,
                )
                selected = np.flatnonzero(pair_distances <= radius)
                if len(selected) == 0:
                    continue
                selected = selected[
                    np.argsort(pair_distances[selected], kind="stable")
                ][:max_pairs]
                pair_features = self._build_pair_features(
                    tracks[pair_indices[0][selected]],
                    tracks[pair_indices[1][selected]],
                )
                trk_pairs_array[
                    event_index, jet_index, :len(pair_features)
                ] = pair_features

        self.jet_matched_trk_pairs_array = trk_pairs_array
        return trk_pairs_array, event_array, jets_array

    def get_split_dataset(self, val_fraction, cut = "1") -> np.array:
        trk_array, event_array, jets_array = self.get_npy_arrays(cut)
        nb_events = trk_array.shape[0]
        val_nb_events = int(np.round(val_fraction * nb_events, decimals = 0))

        val_trk_array = trk_array[:val_nb_events]
        train_trk_array = trk_array[val_nb_events:]

        val_event_array = event_array[:val_nb_events]
        train_event_array = event_array[val_nb_events:]

        return train_trk_array, train_event_array, val_trk_array, val_event_array

    def get_jet_matched_split_dataset(
        self,
        val_fraction,
        cut="1",
        max_jets=6,
    ):
        """Split jet-matched tracks, event features, and jets by event."""
        if not 0 <= val_fraction <= 1:
            raise ValueError("val_fraction must be between 0 and 1")
        trk_array, event_array, jets_array = self.get_jet_matched_npy_arrays(
            cut=cut,
            max_jets=max_jets,
        )
        n_val_events = int(np.round(val_fraction * trk_array.shape[0]))
        return (
            trk_array[n_val_events:],
            event_array[n_val_events:],
            jets_array[n_val_events:],
            trk_array[:n_val_events],
            event_array[:n_val_events],
            jets_array[:n_val_events],
        )

    def get_kfold_dataset(self, kfolds, cut = "1", max_pairs = 10, radius = 0.1) -> np.array:

        if max_pairs is not None:
            trk_array, event_array, jets_array, trk_pairs_array = self.get_track_pairs(
                max_pairs = max_pairs,
                radius = radius,
                cut = cut,
                )
        else:
            trk_array, event_array, jets_array = self.get_npy_arrays(cut)
            trk_pairs_array = None

        nb_events = trk_array.shape[0]

        # Get folds indices
        indices = np.arange(nb_events)
        folds_idx = indices % kfolds

        return [
            (
                trk_array[folds_idx == i],
                event_array[folds_idx == i],
                None if jets_array is None else jets_array[folds_idx == i],
                None if trk_pairs_array is None else trk_pairs_array[folds_idx == i]
                ) for i in range(kfolds)
            ]

    def get_jet_matched_kfold_dataset(
        self,
        kfolds,
        cut="1",
        max_jets=6,
        max_pairs=10,
        radius=0.1,
    ):
        """Return event-aligned jet-matched data for each deterministic fold."""
        if kfolds <= 0:
            raise ValueError("kfolds must be positive")

        if max_pairs is None:
            trk_array, event_array, jets_array = self.get_jet_matched_npy_arrays(
                cut=cut,
                max_jets=max_jets,
            )
            trk_pairs_array = None
        else:
            trk_pairs_array, event_array, jets_array = self.get_jet_matched_track_pairs(
                radius=radius,
                max_pairs=max_pairs,
                cut=cut,
                max_jets=max_jets,
            )
            trk_array = self.jet_matched_trk_array

        fold_indices = np.arange(trk_array.shape[0]) % kfolds
        return [
            (
                trk_array[fold_indices == fold],
                event_array[fold_indices == fold],
                jets_array[fold_indices == fold],
                None if trk_pairs_array is None else trk_pairs_array[fold_indices == fold],
            )
            for fold in range(kfolds)
        ]

    def save_arrays(self, filepath):
        print(
            self.trk_array.shape,
            self.event_array.shape,
            self.jets_array.shape,
            np.array(self.trk_array).shape,
              )

        np.savez_compressed(
            filepath,
            trk_array= self.trk_array,
            event_array= self.event_array,
            jets_array= self.jets_array if self.jets_array is not None else np.array([]),
            trk_pairs_array= self.trk_pairs_array if self.trk_pairs_array is not None else np.array([]),
        )
    
    def load_arrays(self, filepath):
        data = np.load(filepath)
        self.trk_array = data["trk_array"]
        self.event_array = data["event_array"]
        self.jets_array = data.get("jets_array", None)
        self.trk_pairs_array = data.get("trk_pairs_array", None)
        return self.trk_array, self.event_array, self.jets_array, self.trk_pairs_array
        
    # Transform the data to be in range [-1,1]
    def get_lin_transform(self):
        if self.trk_array is None or self.event_array is None:
            raise ValueError("Arrays not computed yet. Call get_npy_arrays() first.")
        
        trk_min = np.min(self.trk_array, axis=(0,1))
        trk_max = np.max(self.trk_array, axis=(0,1))

        event_min = np.min(self.event_array, axis=0)
        event_max = np.max(self.event_array, axis=0)

        trk_shift = (trk_max + trk_min) / 2
        trk_scale = (trk_max - trk_min) / 2
        event_shift = (event_max + event_min) / 2
        event_scale = (event_max - event_min) / 2
        
        return trk_shift, trk_scale, event_shift, event_scale

        
if __name__ == "__main__":

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

    root_file = ROOT.TFile.Open("Multijet_2010B.root")
    tree = root_file.Get("analyzer/Events")
    DP = DataProcessor(
        tree,
        trk_columns,
        event_columns,
        variables_to_define,
        max_events = 5000,
        )
    
    data = DP.get_split_dataset(0.2, )
    DP.get_lin_transform()