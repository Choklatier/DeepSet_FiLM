import numpy as np
import ROOT
from tqdm import tqdm

class DataProcessor:

    def __init__(
        self, 
        tree : ROOT.TTree, 
        trk_columns : list,
        event_columns : list,
        variables_to_define : dict = None,
        max_tracks : int = 50,
        max_events : int = None,
        ) -> None:

        self.trk_columns = trk_columns
        self.event_columns = event_columns
        self.max_tracks = max_tracks

        self.tree = tree
        print(f"Tree has {tree.GetEntries()} entries.")
        # Produce RDF to apply filters, define vars and get npy arrays
        self.rdf = ROOT.RDataFrame(self.tree)
        # Cap max events if asked
        if max_events is not None:
            self.rdf = self.rdf.Range(max_events)

        # Define variables if provided
        if variables_to_define is not None:
            for var in variables_to_define:
                expr = variables_to_define[var]
                self.rdf = self.rdf.Define(var,expr)
        
        # Store last computed arrays
        self.trk_array = None
        self.event_array = None

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

    def get_npy_arrays(self, cut = "1"):
        rdf_filtered = self.rdf.Filter(cut)
        trk_arrays = rdf_filtered.AsNumpy(self.trk_columns)
        event_arrays = rdf_filtered.AsNumpy(self.event_columns)

        # Convert dict of arrays to stricly arrays
        print(f"Preparing {len(self.trk_columns)} track features:")
        tracks_var_arrays = [self.tracks_to_array(trk_arrays[column]) for column in self.trk_columns]
        trk_array_output = np.stack(tracks_var_arrays)

        # Event array are straight forwardly all the same shape
        event_array_output = event_arrays[self.event_columns[0]]
        for column in self.event_columns[1:]:
            event_array_output = np.vstack([event_array_output,event_arrays[column]])

        # Store arrays as fields
        self.trk_array = trk_array_output.T
        self.event_array = event_array_output.T

        return trk_array_output.T, event_array_output.T

    def get_split_dataset(self, val_fraction, cut = "1") -> np.array:
        trk_array, event_array = self.get_npy_arrays(cut)
        nb_events = trk_array.shape[0]
        val_nb_events = int(np.round(val_fraction * nb_events, decimals = 0))

        val_trk_array = trk_array[:val_nb_events]
        train_trk_array = trk_array[val_nb_events:]

        val_event_array = event_array[:val_nb_events]
        train_event_array = event_array[val_nb_events:]

        return train_trk_array, train_event_array, val_trk_array, val_event_array

    def get_kfold_dataset(self, kfolds, cut = "1") -> np.array:
        trk_array, event_array = self.get_npy_arrays(cut)
        nb_events = trk_array.shape[0]
        
        # Get folds indices
        indices = np.arange(nb_events)
        folds_idx = indices % kfolds

        return [(trk_array[folds_idx == i],event_array[folds_idx == i]) for i in range(kfolds)]
        
    def save_arrays(self, filepath):
        ...
    
    def load_arrays(self, filepath):
        ...

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