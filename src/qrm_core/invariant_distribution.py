import numpy as np
from .intensity import IntensityTable

def compute_invariant_distribution(
        side: float,
        intensity_table: IntensityTable,
        dump_path='invariant_distribution.npy'
    ):
    
    """
        Compute and save the invariant distributions π_i for each depth i.
        Refer to Section 2.3.3 of Huang et al. (2015).

        Inputs:
            - side: 'bid', 'ask' or None.
                We allow bid-ask asymmetry in the order flow intensities.
                None means we assume symmetry.  
            - intensity_table: IntensityTable object with order-flow intensities.
            - dump_path: path to .npy file to save the invariant distribution.
        
        Outputs:
            - None. The invariant distribution is saved in a .npy file.
    """


    if side not in [None, 'bid', 'ask']:
        raise ValueError("side must be 'bid', 'ask' or None")
    elif side in [None, 'bid']:
        intensities = intensity_table._data[:, :, 0, :]
    elif side == 'ask':
        intensities = intensity_table._data[:, :, 1, :]

    K, Q1, *_ = intensities.shape         
    Q = Q1 - 1                         
    type_to_index = intensity_table._type_index  
    all_pi = np.zeros((K, Q + 1)) # invariant distribution for each depth           

    for i in range(K):
        # Arrival/departure ratio vector ρ_i (see paper)
        lamL = intensities[i][:-1, type_to_index['limit']]
        lamC = intensities[i][1:, type_to_index['cancel']]
        lamM = intensities[i][1:, type_to_index['market']]
        rho =  lamL / (lamC + lamM)

        pi = np.zeros(Q+1)
        pi_0 = 1 / (1 + np.sum(np.cumprod(rho)))
        pi[0] = pi_0
        pi[1:] = pi_0 * np.cumprod(rho)
        print('sum pi (not normalized)', np.sum(pi)) # monitor if sum is close enough to 1 for a given Q. 
        pi /= np.sum(pi)                   
        all_pi[i] = pi

    folder_path = 'calibration_data/'
    if side in ['bid', 'ask']:
        file_path = dump_path[:-4] + '_' + side + dump_path[-4:]
        np.save(folder_path + file_path, all_pi)
    else:
        np.save(folder_path + dump_path, all_pi)