import numpy as np

LIGHT_SPEED = 299792458*1e3 # mm/s

# LLPMaker produce fake tracks associated with a neutral particle decaying to fermions and associated with a jet.
# Assume jets_array := [pt,eta,phi,e]
class LLPMaker:

    def __init__(
            self, 
            trk_array : np.array, 
            jets_array : np.array,
            trk_columns : list,
            ):
        self.trk_array = trk_array
        self.jets_array = jets_array
        self.trk_columns = trk_columns


    def sample_LLP_tracks_loop(
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

        # Get number of events and first jets
        N_events = len(self.trk_array)
        jets = self.jets_array[:, 0] # highest pT jet
        
        # Get jet kinematics
        pt_jet  = jets[:, 0]
        eta_jet = jets[:, 1]
        phi_jet = jets[:, 2]
        theta = 2*np.arctan(np.exp(-eta_jet))
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi_jet)
        cos_phi = np.cos(phi_jet)
        # Compute jet direction vector
        n_j = np.stack([
            sin_theta*cos_phi,
            sin_theta*sin_phi,
            cos_theta
        ], axis=1)
        # Compute perpendicular vectors
        e_theta = np.stack([
            cos_theta*cos_phi,
            cos_theta*sin_phi,
            -sin_theta
        ], axis=1)
        e_phi = np.stack([
            -sin_phi,
            cos_phi,
            np.zeros(N_events)
        ], axis=1)

        # Sample LLP kinematics
        z = np.random.beta(alpha, beta, N_events)
        sigma = sig_0 * (1 - z)**a
        opening = np.random.rayleigh(sigma)
        dphi = np.random.uniform(0, 2*np.pi, N_events)

        # Compute direction of the resonance
        n_res = (
            np.cos(opening)[:,None]*n_j
            + np.sin(opening)[:,None] *
            (
                np.cos(dphi)[:,None]*e_theta
                + np.sin(dphi)[:,None]*e_phi
            )
        )

        # Get angles of resonance
        theta_res = np.arccos(n_res[:,2])
        eta_res = -np.log(np.tan(theta_res/2))
        phi_res = np.arctan2(
            n_res[:,1],
            n_res[:,0]
        )

        # Compute the resonance energy/momentum based on z
        pt_res = z * pt_jet
        px = pt_res*np.cos(phi_res)
        py = pt_res*np.sin(phi_res)
        pz = pt_res*np.sinh(eta_res)

        p = np.stack([px,py,pz], axis=1)
        pmag = np.linalg.norm(p, axis=1)
        energy = np.sqrt(pmag**2 + mass**2)

        # Use lifetime to get track origins
        proper_time = np.random.exponential(
            lifetime,
            N_events
        )
        distance = (
            pmag/mass
        )*LIGHT_SPEED*proper_time

        vertex = distance[:,None] * p / pmag[:,None]

        # Sample two-body decay
        costheta = np.random.uniform(-1,1,N_events)
        sintheta = np.sqrt(1-costheta**2)
        phi_decay = np.random.uniform(0,2*np.pi,N_events)

        if different_fermion_flavors:
            raise ValueError("Different fermion flavors not implemented yet")
        else:
            # Get fermion mass (e or mu)
            mass_fermion = np.where(
                np.random.rand(N_events) < 0.5,
                0.000511,   # GeV
                0.105658    # GeV
            )
            
            # Momentum magnitude of fermion in rest frame
            pstar = np.sqrt(
                mass**2/4
                - mass_fermion**2
            ) 
            
            # Convert to 3D assuming isotropic decay
            prest = pstar[:, None]*np.stack([
                sintheta*np.cos(phi_decay),
                sintheta*np.sin(phi_decay),
                costheta
            ], axis=1)

            # Now we need to boost back in Lab frame
            beta_res = p/energy[:,None]
            beta_res2 = np.sum(beta_res**2, axis=1)
            gamma = 1/np.sqrt(1-beta_res2)
            Erest = np.sqrt(
                pstar**2
                + mass_fermion**2
            )
            dot = np.sum(
                beta_res*prest,
                axis=1
            )

            boost = (
            ((gamma-1)/beta_res2)[:,None]
            *dot[:,None]
            *beta_res
            +
            (gamma*Erest)[:,None]
            *beta_res
            )

            p1 = prest + boost
            p2 = -prest + (
                ((gamma-1)/beta_res2)[:,None]
                *(-dot)[:,None]
                *beta_res
                +
                (gamma*Erest)[:,None]
                *beta_res
            )

            charge_fermion1 = np.where(
                np.random.rand(N_events) < 0.5,
                -1,
                1    
            )
            charge_fermion2 = -charge_fermion1 if opposite_charge_fermions else np.where(
                np.random.rand(N_events) < 0.5,
                -1,
                1    
            )

            if debug_return:
                return p1, p2, vertex, n_j, p



    def save_arrays(self, filepath : str):
        ...

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
        max_events = 17,
        )
    
    trk_array, event_array, jets_array = DP.get_npy_arrays()

    mass = 100.0 # GeV
    llp_maker = LLPMaker(trk_array, jets_array, trk_columns)
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
    event_nb = 0
    p_fermion1, p_fermion2, d, n_j, p = (
        p_fermion1[event_nb,:], 
        p_fermion2[event_nb,:], 
        d[event_nb,:], n_j[event_nb,:], p[event_nb,:]
    )
    print(
        p_fermion1.shape, p_fermion2.shape,
        d.shape, n_j.shape, p.shape
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


            



        

        