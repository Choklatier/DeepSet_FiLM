import numpy as np

LIGHT_SPEED = 299792458*1e3 # mm/s

# LLPMaker produce fake tracks associated with a neutral particle decaying to fermions and associated with a jet.
# Assume jets_array := [pt,eta,phi,e]
class LLPMaker:

    def __init__(self, trk_array : np.array, jets_array : np.array):
        self.trk_array = trk_array
        self.jets_array = jets_array

    def sample_LLP_tracks(
            self, 
            mass : float, 
            lifetime : float, 
            alpha : float, # Shape parameter for the beta dist (used to sample the momentum of the resonance)
            beta : float, # Shape parameter for the beta dist (used to sample the momentum of the resonance)
            sig_0 : float, # for input to Rayleigh distribution for the transverse momentum
            a : float, # for input to Rayleigh distribution for the transverse momentum
            opposite_charge_fermions : bool = True,
            different_fermion_flavors : bool = False,
            debug_return : bool = False
            ):
        """
        Sample a resonance with a given mass and lifetime, and produce two tracks associated with it.
        """
        
        # TODO : remove for loop and use array logic
        for event_idx in range(len(self.trk_array)):
            # Pick highest jet in the event
            jet = self.jets_array[event_idx][0]
        
            # Sample the momentum of the resonance using the jet        
            z = np.random.beta(alpha, beta)

            # Compute the dR relative to the jet
            sig = sig_0 * (1 - z)**a
            dR = np.random.rayleigh(sig) # Small angle approximation, valid for small dR

            # Sample the azimuthal angle of the resonance relative to the jet
            dphi = np.random.uniform(0, 2 * np.pi)

            # Compute the momentum of the resonance
            # Pt is straightfoward from z
            pt_resonance = z * jet[0]
            # convert to (theta, phi) sphere
            jet_theta = 2 * np.arctan(np.exp(-jet[1]))
            jet_phi = jet[2]
            # direction on the sphere
            n_j = np.array([np.sin(jet_theta) * np.cos(jet_phi), np.sin(jet_theta) * np.sin(jet_phi), np.cos(jet_theta)])
            # Compute perpendicular directions
            n_theta = np.array([np.cos(jet_theta) * np.cos(jet_phi), np.cos(jet_theta) * np.sin(jet_phi), -np.sin(jet_theta)])
            n_phi = np.array([-np.sin(jet_phi), np.cos(jet_phi), 0])
            # direcion of the resonance
            n_resonance = np.cos(dR) * n_j + np.sin(dR) * (np.cos(dphi) * n_theta + np.sin(dphi) * n_phi)
            # convert back to (eta, phi)
            eta = -np.log(np.tan(0.5 * np.arccos(n_resonance[2])))
            phi = np.arctan2(n_resonance[1], n_resonance[0])
            p = np.array([pt_resonance * np.cos(phi), pt_resonance * np.sin(phi), pt_resonance * np.sinh(eta)])

            # Decay the resonance to two fermions
            # Get lifetime in restframe
            proper_time = np.random.exponential(lifetime)
            # Boost to lab frame, TODO : Shift by PV coordinates +  add correction for magnetic field? (d0 is shortest distance with respect to full helix)
            d = (p/mass) * LIGHT_SPEED * proper_time
            d0 = np.sqrt(d[0]**2 + d[1]**2)
            z0 = d[2]

            # Sample the momentum of the fermions in the rest frame of the resonance
            # Assume isotropic decay
            costheta = np.random.uniform(-1, 1)
            sintheta = np.sqrt(1 - costheta**2)
            phi_fermion = np.random.uniform(0, 2 * np.pi)
            # Attribute the momentum of the fermions in the rest frame of the resonance
            # Use the two body decay formula to get the momentum of the fermions in the rest frame of the resonance
            if different_fermion_flavors:
                raise ValueError("Different fermion flavors not implemented yet")
            else:
                mass_fermion = np.random.choice([0.511, 105.658])/1000 # m_e, m_mu GeV
                p_mag_fermion_rest = np.sqrt(mass**2/4 - mass_fermion**2)
                p_fermion_rest = p_mag_fermion_rest * np.array([sintheta * np.cos(phi_fermion), sintheta * np.sin(phi_fermion), costheta])
                E_fermion_rest = np.sqrt(mass_fermion**2 + p_mag_fermion_rest**2)
                # Boost the fermions to the lab frame
                v_resonance = p / np.sqrt(mass**2 + np.sum(p**2))
                v_resonance2 = np.sum(v_resonance**2)
                gamma = 1 / np.sqrt(1 - v_resonance2)
                # TODO : Check below is correct
                p_fermion1 = p_fermion_rest + (gamma - 1) * np.dot(p_fermion_rest, v_resonance) / v_resonance2 * v_resonance + gamma * E_fermion_rest * v_resonance
                p_fermion2 = -p_fermion_rest + (gamma - 1) * np.dot(-p_fermion_rest, v_resonance) / v_resonance2 * v_resonance + gamma * E_fermion_rest * v_resonance

                # TODO : Implement smearing of d0 and z0 based on the momentum of the fermions and the resolution of the detector

            charge_fermion1 = np.random.choice([-1, 1])
            charge_fermion2 = -charge_fermion1 if opposite_charge_fermions else np.random.choice([-1, 1])

            if debug_return:
                return p_fermion1, p_fermion2, d, n_j, p
            
            # TODO :Fill the tracks array with the new tracks

if __name__ == "__main__":
    import ROOT
    from DataProcessor import DataProcessor
    import matplotlib.pyplot as plt

    # Read some data
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
        jets_columns = ["jet_pt", "jet_eta", "jet_phi", "jet_mass"],
        variables_to_define = variables_to_define,
        max_events = 2,
        )
    
    trk_array, event_array, jets_array = DP.get_npy_arrays()

    mass = 100.0 # GeV
    llp_maker = LLPMaker(trk_array, jets_array)
    p_fermion1, p_fermion2, d, n_j, p = llp_maker.sample_LLP_tracks(
        mass=mass,
        lifetime=1e-11,
        alpha=2.0,
        beta=5.0,
        sig_0=0.1,
        a=1.0,
        opposite_charge_fermions=True,
        different_fermion_flavors=False,
        debug_return = True
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    print("resonance p4:",[np.sqrt(np.sum(p**2) + mass), p[0], p[1], p[2]])
    print("p_fermion1 + p_fermion2:", p_fermion1 + p_fermion2)
    print("p_fermion1:",p_fermion1)
    print("p_fermion2:",p_fermion2)
    print("d:",d)
    
    vectors = {
        "p_fermion1": np.asarray(p_fermion1)/np.sqrt(np.sum(p_fermion1**2)),
        "p_fermion2": np.asarray(p_fermion2)/np.sqrt(np.sum(p_fermion2**2)),
        "d" : np.asarray(d),
        "n_j": np.asarray(n_j),
    }

    for (name, vec), color in zip(vectors.items(), ["green", "green", "black", "red"]):
        if "p_fermion" not in name:
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, label=name, arrow_length_ratio=0.1)
        else:
            ax.quiver(d[0], d[1], d[2], vec[0], vec[1], vec[2], color=color, label=name, arrow_length_ratio=0.1)

    ax.set_title("LLP decay geometry")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(min(-0.1,d[0]* 1.2),max(0.1,d[0] * 1.2))
    ax.set_ylim(min(-0.1,d[1]* 1.2),max(0.1,d[1] * 1.2))
    ax.set_zlim(min(-0.1,d[2]* 1.2),max(0.1,d[2] * 1.2))
    ax.legend()
    plt.show()


            



        

        