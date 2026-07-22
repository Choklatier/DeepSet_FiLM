import ROOT
from DataProcessor import DataProcessor
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.special import beta
from scipy.stats import crystalball

plt.rcParams['text.usetex'] = True

"""
The purpose of this script is to fit distributions to track vs jet data.
The fit results will be used to generate LLP resonance with 'realistic' kinematics.
The fits are performed on the highest pT track with respect to its jet.
"""

def sigma_func(x,sig0,a):
    return sig0 * (1 - x)**a

def beta_dist(x,a,b):
    return (1/beta(a,b)) * x**(a - 1) * (1 - x)**(b-1)

def crystalball_dist(x,beta,m,mu,sig):
    rv = crystalball(beta,m,loc = mu, scale = sig)
    return rv.pdf(1-x)

def rayleigh_dist(x,z,sig0,a):
   sigma = sig0 * (1 - z)**a
   return (x/sigma**2) * np.exp(-x**2/(2 * sigma**2))

def weibull_dist(x,z,sig0,a,k):
    sigma = sig0 * (1 - z)**a
    l = np.sqrt(2) * sigma
    return (k/l) * (x/l)**(k-1) * np.exp(-(x/l)**k)

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
    max_events = 100000,
    variables_to_define = variables_to_define,
    )

trk_array, event_array, jets_array = DP.get_npy_arrays()

print(trk_array.shape, jets_array.shape)

# Create masks to ignore jetless events
# and events where the first track is not associated with any jet
# and events where the first track is associated with jet_idx > 5
jetless_mask = jets_array[:, 0, 0] == 0
jetIdx_idx = trk_columns.index("trk_jetIdx")
first_trk_nojet_mask = (trk_array[:,0,jetIdx_idx] == -1)
jet_idx_toohigh_mask = (trk_array[:,0,jetIdx_idx] > 5)

first_mask = (
    jetless_mask 
    | first_trk_nojet_mask 
    | jet_idx_toohigh_mask
    )

print(f"Fist mask shape:",first_mask.shape)
print("First mask efficiency : ", np.sum(~first_mask)/first_mask.shape[0])

# Apply first mask
trk_array = trk_array[~first_mask]
jets_array = jets_array[~first_mask]

# Get indices
jet_idx = trk_array[:,0,jetIdx_idx].astype(int)
event_idx = np.arange(len(trk_array))

# Use it to compute final mask to remove jets that are selected but removed
# by a pT cut.
jet_selected_removed_mask = (
    jets_array[event_idx, jet_idx, 0] == 0
)

# get final mask:
second_mask = jet_selected_removed_mask
print(f"Second mask shape:",second_mask.shape)
print("Second mask efficiency : ", np.sum(~second_mask)/second_mask.shape[0])

# Apply per event mask:
trk_array = trk_array[~second_mask,:,:]
jets_array = jets_array[~second_mask,:,:]

# update jet_idx and event_idx after more masking
jet_idx = trk_array[:,0,jetIdx_idx].astype(np.int_)
event_idx = np.arange(jets_array.shape[0])
# get pT ratio with jet and fit:
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
plt.xlabel("$z$")
plt.ylabel("Normalised Events")
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

# Plot the dR distribution
plt.figure()
counts, bins, _ = plt.hist(dR, bins = 50, density = True, histtype="step")
plt.xlabel("$\Delta R$")
plt.ylabel("Normalised Events")
plt.legend()
plt.savefig("plots/trk_jet_dR.pdf")
# Plot the dPhi and dEta histograms
plt.figure()
plt.hist(dPhi, bins = 50, density = True,histtype="step")
plt.xlabel("$\Delta \phi$")
plt.ylabel("Normalised Events")
plt.legend()
plt.savefig("plots/trk_jet_dPhi.pdf")
plt.figure()
plt.hist(dEta, bins = 50, density = True, histtype="step")
plt.xlabel("$\Delta \eta$")
plt.ylabel("Normalised Events")
plt.legend()
plt.savefig("plots/trk_jet_dEta.pdf")

# In LLP maker, dR is sampled with Rayleigh(sig_0 * (1 - z)^a).
# Let's define the likelihood to estimate the best value for 'sig_0' and 'a'.

def nll(params, z, dr):
    sig0, a = params
    sigma = sig0 * (1-z)**a

    # print(
    #     "nll = ",
    #     np.sum(2*np.log(sigma) + dr**2/(2*sigma**2)), 
    #     ", a = ", a, 
    #     ", sig0 = ",sig0,
    #     "min(sigma) = ", np.min(sigma),
    #     "z[argmin(sigma)] = ", z[np.argmin(sigma)],
    #     ", 2log(sigma) = ",np.sum(2 * np.log(sigma)),
    #     ", dR^2/(2sigma^2) = ",np.sum(dr**2/(2*sigma**2)),
    #     )

    return np.sum(
        2*np.log(sigma)
        + dr**2/(2*sigma**2)
    )

def nll_weibull(params, z, dr):
    sig0, a, k = params
    sigma = sig0 * (1-z)**a
    l = np.sqrt(2) * sigma
    # print(
    #     - np.log(k),
    #     np.sum(+ k * np.log(l)),
    #     np.sum(- (k-1) * np.log(dr)),
    #     np.sum(+ (dr/l)**k)
    #     )

    return np.sum(
        - np.log(k)
        + k * np.log(l)
        - (k-1) * np.log(dr)
        + (dr/l)**k
    )

print("Minimising Rayleigh likelihood...")
x0 = [0.1, 1.0] # initial parameters for sig0 and a
res_rayleigh = minimize(
    nll, x0, 
    method='Nelder-Mead', 
    args=(z[z < 1.0], dR[z < 1.0]),
    bounds = [
        (1e-6, None), # sig0 > 0
        (0.0, None), # a >= 0
        ],
    tol=1e-6
    )

print(f"Optimisation {'successful' if res_rayleigh.success else 'failed'}!")
print("Optimisation ending with : ",res_rayleigh.message)
print("Finall nll = ",nll(res_rayleigh.x, z[z < 1.0], dR[z < 1.0]))
print("Final result: [sig0, a] = ",res_rayleigh.x)


print("Minimising Weibull likelihood...")
x0 = [0.1, 1.0, 2.0] # initial parameters for sig0, a and k
res_weibull = minimize(
    nll_weibull, x0, 
    method='Nelder-Mead', 
    args=(z[z < 1.0], dR[z < 1.0]),
    bounds = [
        (1e-6, None), # sig0 > 0
        (0.0, None), # a >= 0
        (1e-6, None), # k > 0
        ],
    tol=1e-6
    )

print(f"Optimisation {'successful' if res_weibull.success else 'failed'}!")
print("Optimisation ending with : ",res_weibull.message)
print("Finall nll = ",nll_weibull(res_weibull.x, z[z < 1.0], dR[z < 1.0]))
print("Final result: [sig0, a, k] = ",res_weibull.x)

# Divide data into z-bins to test the fit
z_bins = np.arange(0.0,1.1,0.1)
dR_lin = np.linspace(np.min(dR), np.max(dR),10000)
sigmas = []
z_centers = []
for i in range(len(z_bins) - 1):
    z_down = z_bins[i]
    z_up = z_bins[i+1]
    z_center = (z_up + z_down)/2
    
    z_cut = (z >= z_down) & (z < z_up)

    print(f"{len(dR[z_cut])} events in bin [{z_down},{z_up}]")

    # Compute sigmas and store
    sigmas.append(
        np.sqrt(
            np.sum(
                dR[z_cut]**2) * (1/(2 * len(dR[z_cut]))
            )
        )
    )
    z_centers.append(z_center)
    
    plt.figure()
    plt.title(f"$z~\in~[{z_down},{z_up}]$")
    plt.hist(dR[z_cut],density = True, label = "data", bins = 20)
    plt.plot(dR_lin,rayleigh_dist(dR_lin,z_center,*res_rayleigh.x), label = "Rayleigh function fit")
    plt.plot(dR_lin,weibull_dist(dR_lin,z_center,*res_weibull.x), label = "Weibull function fit")
    plt.xlabel("$\Delta R$")
    plt.ylabel("Normalised Events")
    plt.legend()
    plt.savefig(f"plots/trk_jet_dR_z_{z_down:0.2f}_{z_up:0.2f}.pdf")

z_centers, sigmas = np.array(z_centers), np.array(sigmas)
sig0_rayleigh, a_rayleigh = res_rayleigh.x
sig0_weibull, a_weibull, k_weibull = res_weibull.x

# Let's do a direct fit to see what we can extract
popt_sig,pcov_sig = curve_fit(
    sigma_func,
    z_centers, sigmas,
    )
print("Fitting (z_centers,sigmas):")
print("Results : [sig0,a] = ",popt_sig)

plt.figure()
plt.scatter(z_centers,sigmas)
plt.plot(
    z_lin, sig0_rayleigh * (1-z_lin)**a_rayleigh,
    label = "Extracted from Rayleigh distribution fit"
    )
plt.plot(
    z_lin, sig0_weibull * (1-z_lin)**a_weibull,
    label = "Extracted from Weibull distribution fit"
    )
plt.plot(z_lin, sigma_func(z_lin,*popt_sig),label = "Direct powerlaw fit")
plt.xlabel("$z_{centers}$")
plt.ylabel("$\sigma(z)$")
plt.grid()
plt.legend()
plt.savefig("plots/z_sigma.pdf")