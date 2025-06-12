import numpy as np
from src.qrm_core.intensity import IntensityTable

def compute_invariant_distribution(
        side: float,
        intensity_table: IntensityTable,
        dump_path: str
    ):
    """
        Precompute and store the invariant distributions $π_i(n)$ for each queue depth $i$ in an array.
        Refer to Section 2.3.3 of Huang et al. (2015).

    Inputs:
        - side: 'bid', 'ask' or None. We allow bid-ask asymmetry and thus two different invariant distributions.
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

    K, Q_plus_1, *_ = intensities.shape           # max depth, max queue size + 1, number of types
    Q = Q_plus_1 - 1                              # max queue size (in units of AES)
    type_to_index = intensity_table._type_index   # type -> index
    all_pi = np.zeros((K, Q + 1))                 # invariant distribution for each depth (bid-ask symmetry)

    for i in range(K):
        # Arrival/departure ration vector $\rho_i$
        lamL = intensities[i][:-1, type_to_index['limit']]
        lamC = intensities[i][1:, type_to_index['cancel']]
        lamM = intensities[i][1:, type_to_index['market']]
        rho =  lamL / (lamC + lamM)

        pi = np.zeros(Q+1)
        pi_0 = 1 / (1 + np.sum(np.cumprod(rho)))
        pi[0] = pi_0
        pi[1:] = pi_0 * np.cumprod(rho)
        print('sum pi not normalized', np.sum(pi)) # not exactly 1 as we limit the queue size to maximum Q
        pi /= np.sum(pi)                           # normalize
        all_pi[i] = pi

    folder_path = 'calibration_data/invariant_distribution/'
    if side in ['bid', 'ask']:
        file_path = dump_path[:-4] + '_' + side + dump_path[-4:]
        np.save(folder_path + file_path, all_pi)
    else:
        np.save(folder_path + dump_path, all_pi)