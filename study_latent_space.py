import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, roc_auc_score

from matplotlib.lines import Line2D
plt.rcParams['text.usetex'] = True

from DataProcessor import DataProcessor
from LLPMaker import LLPMaker

import onnx
import onnxruntime as ort

import yaml
"""
This script aims at studying the latent space of the DeepSet+FiLM VAE network.
The model, saved as an onnx format, is imported and tested with different inputs.
A main interest is to see the effect of injecting fake tracks that looks like LLPs.
"""

# Read settings from the config
with open("config.yaml","r") as config_file:
    config = yaml.safe_load(config_file)

# Read input information from yaml config
variables_to_define = config["Inputs"]["variables_to_define"]
trk_columns = config["Inputs"]["trk_columns"]
event_columns = config["Inputs"]["event_columns"]

MAX_TRACKS = config["Architecture"]["MAX_TRACKS"]
RHO_DIM = config["Architecture"]["RHO_DIM"]
LATENT_DIM = config["Architecture"]["LATENT_DIM"]

MAX_EVENTS = config["Training"]["MAX_EVENTS"]
K_FOLDS    = config["Training"]["K_FOLDS"]

# Useful functions
def plot_gauss_smooth_contour(x,y, label = "", levels = 5, color = "black"):
    H, xedges, yedges = np.histogram2d(x, y, bins=100, density=True)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)
    H = gaussian_filter(H, sigma=2)
    plt.contour(
        X, Y, H.T, 
        label = label, 
        levels = levels, 
        colors = color
        )

# Computes the square of the mahalanobis distance for a vector z
def mahalanobis2(z):
    mu = np.mean(z,axis = 0)
    Sigma = np.cov(z, rowvar=False)
    Sigma_inv = np.linalg.pinv(Sigma)

    dz = z - mu
    return np.einsum("...i,ij,...j->...", dz, Sigma_inv, dz)

# creates an empirical p-value from score distribution (from empirical cdf)
# reference is the sorted scores we are estimating the cdf of
def empirical_pvalue(score, reference):
    idx = np.searchsorted(reference, score, side="right")
    return 1.0 - idx / len(reference)

# Define s(z)
def compute_score(DM, reference):
    return -np.log(
        np.maximum(
            empirical_pvalue(DM,reference),
            1e-12
            )
        )

# Prepare some input data
print("Reading data...")
root_file = ROOT.TFile.Open("Multijet_2010B.root")
tree = root_file.Get("analyzer/Events")
print("Processing data...")
DP = DataProcessor(
    tree,
    trk_columns,
    event_columns,
    jets_columns = ["jet_pt", "jet_eta", "jet_phi", "jet_mass"],
    variables_to_define = variables_to_define,
    max_events = MAX_EVENTS,
    max_tracks = MAX_TRACKS,
    )
print("Dividing data into folds...")
folds = DP.get_kfold_dataset(kfolds = K_FOLDS, cut = "1")

val_trk_array, val_event_array, val_jets_array = folds[0]
train_trk_array, train_event_array, train_jets_array = folds[1]

# Create the masks
val_valid_tracks = np.minimum(val_event_array[:, 2].astype(np.int32), MAX_TRACKS)
val_mask = (
    np.arange(MAX_TRACKS)[None, :] < val_valid_tracks[:, None]
).astype(np.float32)
val_mask = val_mask[..., None]

train_valid_tracks = np.minimum(train_event_array[:, 2].astype(np.int32), MAX_TRACKS)
train_mask = (
    np.arange(MAX_TRACKS)[None, :] < train_valid_tracks[:, None]
).astype(np.float32)
train_mask = train_mask[..., None]

# Convert data to float32
train_trk_array = train_trk_array.astype(np.float32, copy=False)
train_event_array = train_event_array.astype(np.float32, copy=False)
val_trk_array = val_trk_array.astype(np.float32, copy=False)
val_event_array = val_event_array.astype(np.float32, copy=False)
val_mask = val_mask.astype(np.float32, copy=False)

# Inject some LLPs
llp_maker = LLPMaker(val_trk_array, val_jets_array, trk_columns)
llp_trk_array = llp_maker.sample_LLP_tracks(
        mass=100, # GeV
        lifetime=1e-9, # seconds
        alpha=2.0,
        beta=5.0,
        sig_0=0.1,
        a=1.0,
        opposite_charge_fermions=True,
        different_fermion_flavors=False,
        debug_return = False
    )

# Load model with ONNX to check it
onnx_model = onnx.load("deepset_film.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX checked!")

print(llp_trk_array, val_trk_array)
print(llp_trk_array.shape, val_trk_array.shape)

# Setup a session for inference
sess = ort.InferenceSession("deepset_film.onnx")
val_outputs = sess.run(
    None, {
        'tracks': val_trk_array,
        'mask': val_mask,
        'event': val_event_array,
        }
    )
train_outputs = sess.run(
    None, {
        'tracks': train_trk_array,
        'mask': train_mask,
        'event': train_event_array,
        }
    )
llp_outputs = sess.run(
    None, {
        'tracks': llp_trk_array,
        'mask': val_mask,
        'event': val_event_array,
        }
    )

val_rho = val_outputs[0]
train_rho = train_outputs[0]
llp_rho = llp_outputs[0]
for i in range(RHO_DIM):
    for j in range(RHO_DIM):
        if i >= j : continue

        val_ri = val_rho[::100,i]
        val_rj = val_rho[::100,j]
        val_ri_mean = np.mean(val_rho[:,i], axis = 0)
        val_rj_mean = np.mean(val_rho[:,j], axis = 0)

        train_ri = train_rho[::100,i]
        train_rj = train_rho[::100,j]
        train_ri_mean = np.mean(train_rho[:,i], axis = 0)
        train_rj_mean = np.mean(train_rho[:,j], axis = 0)

        llp_ri = llp_rho[::100,i]
        llp_rj = llp_rho[::100,j]
        llp_ri_mean = np.mean(llp_rho[:,i], axis = 0)
        llp_rj_mean = np.mean(llp_rho[:,j], axis = 0)

        print(f"For r{i} - r{j} means:")
        print(
            "|val - train| = ",
            np.sqrt(
                (val_ri_mean - train_ri_mean)**2 + (val_rj_mean - train_rj_mean)**2
            ) / np.sqrt(val_ri_mean**2 + val_rj_mean**2)
            )
        print(
            "|val - llp| = ",
            np.sqrt(
                (val_ri_mean - llp_ri_mean)**2 + (val_rj_mean - llp_rj_mean)**2
            ) / np.sqrt(val_ri_mean**2 + val_rj_mean**2)
            )

        plt.figure()
        val_sc = plt.scatter(val_ri,val_rj, label = "validation")
        train_sc = plt.scatter(train_ri,train_rj, label = "training")
        llp_sc = plt.scatter(llp_ri,llp_rj, label = "llp injected")

        # print means
        plt.scatter(
            val_ri_mean,val_rj_mean,
            marker="D",
            color = val_sc.get_facecolors()[0],
            edgecolor = "black",
            )
        plt.scatter(
            train_ri_mean,train_rj_mean,
            marker="D",
            color = train_sc.get_facecolors()[0],
            edgecolor = "black",
            )
        plt.scatter(
            llp_ri_mean,llp_rj_mean,
            marker="D",
            color = llp_sc.get_facecolors()[0],
            edgecolor = "black",
            )

        # Add custom legend entry
        plt.legend()
        custom = Line2D(
            [], [],
            marker='D',
            color='none',
            markerfacecolor='white',
            label=fr'Average point: ($\langle r_{i} \rangle$,$ \langle r_{j} \rangle$)'
        )
        legend = plt.gca().get_legend()
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.get_texts()]
        handles.append(custom)
        labels.append(fr'Average point: ($\langle r_{i} \rangle$,$ \langle r_{j} \rangle$)')

        plt.title(f"$r_{i}$ - $r_{j}$")
        plt.xlabel(f"$r_{i}$")
        plt.ylabel(f"$r_{j}$")
        plt.legend(handles,labels)
        plt.savefig(f"plots/latent_space_study/rhodim{RHO_DIM}_r{i}_r{j}.pdf")

        plt.figure()
        plot_gauss_smooth_contour(
            val_rho[:,i],val_rho[:,j],
            label = "validation event distribution",
            color = "blue"
            )
        plot_gauss_smooth_contour(
            llp_rho[:,i],llp_rho[:,j],
            label = "llp-injected event distribution",
            color = "green"
        )
        plt.title(f"$r_{i}$ - $r_{j}$")
        plt.xlabel(f"$r_{i}$")
        plt.ylabel(f"$r_{j}$")
        plt.savefig(f"plots/latent_space_study/rhodim{RHO_DIM}_gauss_r{i}_r{j}.pdf")
        plt.close("all")


# Compute s(z) for both val and llp
DM_val = mahalanobis2(val_rho)
DM_llp = mahalanobis2(llp_rho)
DM_val_sorted = np.sort(DM_val) 

score_val = compute_score(DM_val, DM_val)
score_llp = compute_score(DM_llp, DM_llp)

# Plot s(z) distributions
plt.figure()
plt.hist(score_val, bins = 30 , histtype="step",label = "validation")
plt.hist(score_llp, bins = 30 , histtype="step",label = "llp-injected")
plt.savefig("plots/latent_space_study/score_distributions.pdf")

# AUC of s(z)
y_pred = np.concatenate([score_llp,score_val],axis = 0)
y_true = np.concatenate([np.ones(score_llp.shape),np.zeros(score_val.shape)],axis = 0)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.figure()
plt.title(f"AUC = {roc_auc_score(y_true, y_pred)}")
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],ls = "--",color = "grey")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.savefig("plots/latent_space_study/score_ROC.pdf")