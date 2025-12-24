import numpy as np 
import pandas as pd

from qrm_core.engine import simulate_QRM_jit
from qrm_core.sampling import sample_stationary_lob
from qrm_core.intensity import IntensityTable


def to_dataframe(times, p_mids, p_refs, sides, depths, events, redrawn, states, K):
    """
        Convert logged LOB simulation data (numpy) to pandas DataFrame.
    """
    side_labels  = np.array(['', 'bid', 'ask'], dtype=object)
    event_labels = np.array(['', 'limit', 'cancel', 'market', 'trader'], dtype=object)
    side_cat  = pd.Categorical(side_labels[sides],  categories=['bid','ask'])
    event_cat = pd.Categorical(event_labels[events], categories=['limit','cancel','market','trader'])

    df = pd.DataFrame({
        'time':    times,
        'p_mid':   p_mids,
        'p_ref':   p_refs,
        'side':    side_cat,
        'depth':   depths,
        'event':   event_cat,
        'redrawn': redrawn.astype(bool, copy=False),
    }, copy=False)

    q_df = pd.DataFrame(states, copy=False)
    order = list(range(K-1, -1, -1)) + list(range(K, 2*K))
    q_df = q_df.iloc[:, order]

    bid_names = [f"q_bid{i+1}" for i in range(K)][::-1]  # q_bidK ... q_bid1
    ask_names = [f"q_ask{i+1}" for i in range(K)]        # q_ask1 ... q_askK
    q_df.columns = bid_names + ask_names

    out = pd.concat([df, q_df], axis=1, copy=False)
    return out