"""Compare configured physical track features for data and injected LLP tracks."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml

from LLPMaker import LLPMaker
from LLPMaker import LIGHT_SPEED


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
ROOT_PATH = SCRIPT_DIR / "Multijet_2010B.root"
PLOT_DIR = SCRIPT_DIR / "plots" / "llp_tracks"
MOMENTUM_PLOT_RANGES_GEV = {
    "trk_pt": (0.0, 200.0),
    "trk_px": (-200.0, 200.0),
    "trk_py": (-200.0, 200.0),
}


def load_track_data(config):
    """Load configured track and jet arrays from cache or the ROOT input."""
    inputs = config["Inputs"]
    max_events = config["Training"]["MAX_EVENTS"]
    max_tracks = config["Architecture"]["MAX_TRACKS"]

    if inputs["LOAD_ARRAYS"]:
        arrays_path = SCRIPT_DIR / inputs["ARRAYS_FILEPATH"]
        if not arrays_path.is_file():
            raise FileNotFoundError(f"Configured array file not found: {arrays_path}")
        with np.load(arrays_path) as arrays:
            required = {"trk_array", "event_array", "jets_array"}
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(
                    f"Array file {arrays_path} is missing: {', '.join(sorted(missing))}"
                )
            trk_array = arrays["trk_array"][:max_events, :max_tracks]
            event_array = arrays["event_array"][:max_events]
            jets_array = arrays["jets_array"][:max_events]
    else:
        import ROOT

        from DataProcessor import DataProcessor

        if not ROOT_PATH.is_file():
            raise FileNotFoundError(f"ROOT input file not found: {ROOT_PATH}")
        root_file = ROOT.TFile.Open(str(ROOT_PATH))
        if not root_file or root_file.IsZombie():
            raise OSError(f"Could not open ROOT input file: {ROOT_PATH}")
        try:
            tree = root_file.Get("analyzer/Events")
            if not tree:
                raise ValueError(f"Could not find analyzer/Events in {ROOT_PATH}")
            processor = DataProcessor(
                tree=tree,
                trk_columns=inputs["trk_columns"],
                event_columns=inputs["event_columns"],
                jets_columns=inputs["jet_columns"],
                variables_to_define=inputs["variables_to_define"],
                max_tracks=max_tracks,
                max_events=max_events,
            )
            trk_array, event_array, jets_array = processor.get_npy_arrays()
        finally:
            root_file.Close()

    n_features = len(inputs["trk_columns"])
    n_jet_features = len(inputs["jet_columns"])
    if trk_array.ndim != 3 or trk_array.shape[-1] != n_features:
        raise ValueError(
            f"Track array must have shape (events, tracks, {n_features}); "
            f"got {trk_array.shape}"
        )
    if event_array.ndim != 2 or event_array.shape[0] != trk_array.shape[0]:
        raise ValueError("Event and track arrays must contain the same number of events")
    if (
        jets_array.ndim != 3
        or jets_array.shape[0] != trk_array.shape[0]
        or jets_array.shape[-1] != n_jet_features
        or jets_array.shape[1] == 0
    ):
        raise ValueError(
            "Jet array must have shape "
            f"(events, jets, {n_jet_features}) with at least one jet slot; "
            f"got {jets_array.shape}"
        )
    return trk_array, jets_array


def plot_feature_distributions(data_tracks, llp_tracks, feature_names, output_dir):
    """Save a density comparison for each configured track feature."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_valid = np.any(data_tracks != 0, axis=-1)
    llp_valid = np.any(llp_tracks != 0, axis=-1)
    if "trk_eta" not in feature_names:
        raise ValueError("trk_eta must be configured to apply the |eta| <= 2.5 cut")
    eta_index = feature_names.index("trk_eta")
    data_eta = data_tracks[..., eta_index]
    llp_eta = llp_tracks[..., eta_index]
    data_valid &= np.isfinite(data_eta) & (np.abs(data_eta) <= 2.5)
    llp_valid &= np.isfinite(llp_eta) & (np.abs(llp_eta) <= 2.5)

    for feature_index, feature_name in enumerate(feature_names):
        data_values = data_tracks[..., feature_index][data_valid]
        llp_values = llp_tracks[..., feature_index][llp_valid]
        data_values = data_values[np.isfinite(data_values)]
        llp_values = llp_values[np.isfinite(llp_values)]
        if data_values.size == 0:
            raise ValueError(f"No finite data-track values for feature {feature_name}")
        if llp_values.size == 0:
            raise ValueError(f"No finite injected LLP-track values for feature {feature_name}")

        plot_range = MOMENTUM_PLOT_RANGES_GEV.get(feature_name)
        if plot_range is not None:
            lower, upper = plot_range
            data_values = data_values[
                (data_values >= lower) & (data_values <= upper)
            ]
            llp_values = llp_values[
                (llp_values >= lower) & (llp_values <= upper)
            ]
            if data_values.size == 0 or llp_values.size == 0:
                raise ValueError(
                    f"No data and/or LLP tracks for {feature_name} within "
                    f"the plotting range [{lower}, {upper}] GeV"
                )
            bin_edges = np.linspace(lower, upper, 101)
        else:
            bin_edges = np.histogram_bin_edges(
                np.concatenate((data_values, llp_values)),
                bins=100,
            )
        figure, axis = plt.subplots()
        axis.hist(
            data_values,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=1.5,
            label="Normal data tracks",
        )
        axis.hist(
            llp_values,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=1.5,
            label="Injected LLP tracks",
        )
        unit = " [GeV]" if plot_range is not None else ""
        axis.set_xlabel(f"{feature_name}{unit}")
        axis.set_ylabel("Probability density")
        if plot_range is not None:
            axis.set_xlim(plot_range)
            axis.text(
                0.98,
                0.98,
                "Tracks outside this range are omitted",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize="small",
            )
        axis.legend()
        figure.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_name)
        figure.savefig(output_dir / f"{safe_name}.pdf")
        plt.close(figure)


def main():
    with CONFIG_PATH.open() as config_file:
        config = yaml.safe_load(config_file)

    inputs = config["Inputs"]
    track_names = inputs["trk_columns"]
    trk_array, jets_array = load_track_data(config)

    leading_jets = jets_array[:, 0, :]
    has_leading_jet = (
        np.isfinite(leading_jets).all(axis=1)
        & (leading_jets[:, 0] > 0)
    )
    if not np.any(has_leading_jet):
        raise ValueError("No events with a finite, positive-pT leading jet were found")

    data_tracks = trk_array[has_leading_jet]
    selected_jets = jets_array[has_leading_jet]
    llp_maker = LLPMaker(data_tracks, selected_jets, track_names)
    _, injected_tracks = llp_maker.sample_LLP_tracks(
        mass=0.75,
        lifetime=10/LIGHT_SPEED,  # in mm
        alpha=2.0,
        beta=5.0,
        sig_0=0.1,
        a=1.0,
        opposite_charge_fermions=True,
        different_fermion_flavors=False,
        min_lifetime=None,
        n_llps_mean=2.0,
        n_llps_std=0.5,
        d0_smearing_sigma  = 0.0001,
        z0_smearing_sigma  = 0.0005,
    )

    plot_feature_distributions(
        data_tracks,
        injected_tracks,
        track_names,
        PLOT_DIR,
    )
    print(
        f"Plotted {len(track_names)} track features for "
        f"{len(data_tracks)} data events and "
        f"{np.count_nonzero(np.any(injected_tracks != 0, axis=-1))} injected LLP tracks "
        f"in {PLOT_DIR}"
    )


if __name__ == "__main__":
    main()
