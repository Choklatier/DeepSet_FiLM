import ROOT
from DataProcessor import DataProcessor
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.special import beta
from scipy.stats import crystalball

"""
The purpose of this script is to fit distributions to track vs jet data.
The fit results will be used to generate LLP resonance with 'realistic' kinematics.
The fits are performed on the highest pT track with respect to its jet.
"""

def beta_dist(x,a,b):
    return (1/beta(a,b)) * x**(a - 1) * (1 - x)**(b-1)

def crystalball_dist(x,beta,m,mu,sig):
    rv = crystalball(beta,m,loc = mu, scale = sig)
    return rv.pdf(1-x)

# Read some data
variables_to_define = {
        "met_px" : "met_pt * cos(met_phi)",
        "met_py" : "met_pt * sin(met_phi)",
    }

trk_columns = [
    "trk_pt",
    "trk_d0",
    "trk_eta",
    "trk_phi",
    "trk_jetIdx"
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
    max_tracks = 2,
    jets_columns = ["jet_pt", "jet_eta", "jet_phi", "jet_mass"],
    max_events = 200000,
    variables_to_define = variables_to_define,
    )

trk_array, event_array, jets_array = DP.get_npy_arrays()

print(trk_array.shape, jets_array.shape)

# Create masks to ignore jetless events
# and events where the first track is not associated with any jet
# and events where the first track is associated with jet_idx > 5
jetless_mask = np.any(jets_array[:, :, 0] == 0, axis=1)
jetIdx_idx = trk_columns.index("trk_jetIdx")
first_trk_nojet_mask = (trk_array[:,0,jetIdx_idx] == -1)
jet_idx_toohigh_mask = (trk_array[:,0,jetIdx_idx] > 5)
combined_mask = (jetless_mask | first_trk_nojet_mask | jet_idx_toohigh_mask)
print("Combined mask efficiency : ", np.sum(combined_mask)/combined_mask.shape[0])
# Apply per event mask:
trk_array = trk_array[~combined_mask,:,:]
jets_array = jets_array[~combined_mask,:,:]

# get pT ratio with jet and fit:
jet_idx = trk_array[:,0,jetIdx_idx].astype(np.int_)
event_idx = np.arange(jets_array.shape[0])
trk_pt_idx = trk_columns.index("trk_pt")
z = trk_array[:,0,trk_pt_idx]/jets_array[event_idx,jet_idx,0]

z_max = 1.0

print(z)
plt.figure()
counts, bins, _ = plt.hist(z[z <= z_max], bins = 50, density = True, label = "data")
centers = (bins[1:] + bins[:-1])/2
# fit to beta function
popt_beta,pcov_beta = curve_fit(
    beta_dist, 
    centers, counts
    )
popt_crystal,pcov_crystal = curve_fit(
    crystalball_dist, 
    centers, counts,
    bounds = ( # Bounds on Beta and m for crystalball
        np.array([0.0,1.0, -np.inf,-np.inf]),
        np.array([np.inf,np.inf, np.inf,np.inf]),
        )
    )
z_lin = np.linspace(0.0,z_max,10000)
plt.plot(
    z_lin,beta_dist(z_lin,*popt_beta),
    label = "Beta function fit",
    )
plt.plot(
    z_lin,crystalball_dist(z_lin,*popt_crystal),
    label = "Crystalball function fit",
    )
plt.legend()
plt.savefig("plots/trk_jet_pt_ratio.pdf")

print(" Fit Result for the Beta funtion fit:")
errs = np.sqrt(np.diag(pcov_beta))
print("Alpha = ", popt_beta[0], "±", errs[0])
print("Beta = ", popt_beta[1], "±", errs[1])
print("Covariance matrix = ", pcov_beta)
print()

print(" Fit Result for the Crystalball funtion fit:")
errs = np.sqrt(np.diag(pcov_crystal))
print("Beta = ", popt_crystal[0], "±", errs[0])
print("m = ", popt_crystal[1], "±", errs[1])
print("mu = ", popt_crystal[2], "±", errs[2])
print("sig = ", popt_crystal[3], "±", errs[3])
print("Covariance matrix = ", pcov_crystal)
print()

# get opening angle of track with jet and fit:
trk_eta_idx = trk_columns.index("trk_eta")
trk_phi_idx = trk_columns.index("trk_phi")

dEta = trk_array[:,0,trk_eta_idx] - jets_array[event_idx,jet_idx,1]
dPhi = (trk_array[:,0,trk_phi_idx] - jets_array[event_idx,jet_idx,2] + np.pi) % (2*np.pi) - np.pi
dR = np.sqrt(dEta**2 + dPhi**2)

plt.close("all")

plt.figure()
counts, bins, _ = plt.hist(dR, bins = 50, density = True, label = "data")
plt.legend()
plt.savefig("plots/trk_jet_dR.pdf")
