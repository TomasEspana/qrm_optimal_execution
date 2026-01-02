import numpy as np 
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"



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


def plot_gap_mean(executed, n):
    """
        For each run in `executed` (list of executed volumes),
        compute the mean number of zeros between consecutive non-zero volumes
        (i.e. the “gap” length), group those means by the length of the run,
        and draw a boxplot for each possible run length 1..len(trader_times)-1.
        
        Inputs:
            - executed : dict[int, list[int]]
                Keys are run IDs; values are lists of volumes (integers) executed
                at each trader time step (with zeros).
            - n : int
                Number of trader time steps.
    """
    x_vals = list(range(1, n))
    
    def mean_zero_gap(vols):
        idxs = [i for i, v in enumerate(vols) if v != 0]
        if len(idxs) < 2:
            return None
        # gap = count of zeros between each pair of non-zeros
        gaps = [idxs[i+1] - idxs[i] - 1 for i in range(len(idxs)-1)]
        return sum(gaps) / len(gaps)
    
    grouped = {L: [] for L in x_vals}
    for _, vols in executed.items():
        L = len(vols)
        if L in grouped:
            mg = mean_zero_gap(vols) 
            if mg is not None:
                grouped[L].append(mg)
    
    data   = []
    pos    = []
    for L in x_vals:
        if grouped[L]:
            data.append(grouped[L])
            pos.append(L)
    
    plt.figure(figsize=(5, 4))
    plt.boxplot(data, positions=pos, widths=0.6, showfliers=False, label='No outliers')
    plt.xlabel('Episode length', fontsize=13)
    plt.ylabel('Gap Mean', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(5, 4))
    plt.boxplot(data, positions=pos, widths=0.6, label='With outliers')
    plt.xlabel('Episode length', fontsize=13)
    plt.ylabel('Gap Mean', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.legend()
    plt.tight_layout()
   
    plt.show()



def plot_gap_variance(executed, n):
    """
        For each run in `executed` (list of executed volumes),
        compute the variance of the number of zeros between consecutive non-zero volumes
        (i.e. the “gap” length), group those variances by the length of the run,
        and draw a boxplot for each possible run length 1..len(trader_times)-1.
        
        Inputs:
            - executed : dict[int, list[int]]
                Keys are run IDs; values are lists of volumes (integers) executed
                at each trader time step (with zeros).
            - n : int
                Number of trader time steps.
    """
    x_vals = list(range(1, n))
    
    def var_zero_gap(vols):
        idxs = [i for i, v in enumerate(vols) if v != 0]
        if len(idxs) < 2:
            return None
        # gap = count of zeros between each pair of non-zeros
        gaps = [idxs[i+1] - idxs[i] - 1 for i in range(len(idxs)-1)]
        return np.var(gaps)
    
    grouped = {L: [] for L in x_vals}
    for _, vols in executed.items():
        L = len(vols)
        if L in grouped:
            mg = var_zero_gap(vols) 
            if mg is not None:
                grouped[L].append(mg)
    
    data   = []
    pos    = []
    for L in x_vals:
        if grouped[L]:
            data.append(grouped[L])
            pos.append(L)
    
    plt.figure(figsize=(5, 4))
    plt.boxplot(data, positions=pos, widths=0.6, showfliers=False, label='No outliers')
    plt.xlabel('Episode length', fontsize=13)
    plt.ylabel('Gap Variance', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(5, 4))
    plt.boxplot(data, positions=pos, widths=0.6, label='With outliers')
    plt.xlabel('Episode length', fontsize=13)
    plt.ylabel('Gap Variance', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.tight_layout()
    plt.legend()
    
    plt.show()



def plot_overview(
    run_names,
    run_labels,
    initial_inventory,
    trader_times,
    metric='executed',
    kde_only=True
    ):

    """
        Plots to assess and compare agent performance across different test runs.
    """

    non_red_colors = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    final_is = {}
    actions = {}
    ddqn_lengths = []

    for lbl, rn in zip(run_labels, run_names):
        with open(RESULTS_DIR / f"{rn}.pkl", "rb") as f:
            dic = pickle.load(f)
        final_is[lbl] = np.asarray(dic['final_is'])
        exec_dict = dic[metric]
        actions[lbl] = exec_dict
        if lbl == 'DDQN':
            seqs = [exec_dict[i] for i in sorted(exec_dict)]
            ddqn_lengths = [len(seq) for seq in seqs]

    # x-grids for IS plots
    maxi = max(np.max(np.abs(v)) for v in final_is.values())
    x_sym = np.linspace(-maxi, maxi, 1000)

    # ===============================
    # Figure 1: IS KDE
    # ===============================
    fig1, ax1 = plt.subplots(figsize=(4,3))
    labels_to_plot = final_is.keys()
    if not kde_only:
        for lbl in labels_to_plot:
            vals = final_is[lbl]
            ax1.hist(vals, bins=27, density=True, alpha=0.3, label=f'{lbl} hist')

    for kk, lbl in enumerate(labels_to_plot):
        print(f'[{lbl}] IS: mean={final_is[lbl].mean():.3f}, std={final_is[lbl].std():.3f} \n')
        vals = final_is[lbl]
        kde = gaussian_kde(vals)
        y = kde(x_sym)
        if lbl != 'DDQN':
            ax1.plot(x_sym, y, label=lbl, color=non_red_colors[kk])
        else:
            ax1.plot(x_sym, y, linestyle='--', color='C3', label=lbl)

    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Density')
    extra = 1 # 3 for dim 4
    ax1.set_xlim(x_sym.min() + extra, x_sym.max() - extra)
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()

    # ===============================
    # Figure 2: Avg. Inventory Trajectory (μ ± σ/2)
    # ===============================
    fig2, ax2 = plt.subplots(figsize=(4,3))
    for kk, (lbl, exec_dict) in enumerate(actions.items()):
        seqs = [exec_dict[i] for i in sorted(exec_dict)]
        inv = np.zeros((len(seqs), len(trader_times)))
        for i, seq in enumerate(seqs):
            inv[i, :len(seq)] = seq
        inv = np.cumsum(inv, axis=1) / initial_inventory

        # stats
        mask = inv[:, -1] < 1 # runs that did not reach full inventory
        pct = np.mean(mask) * 100
        if np.any(mask):
            inv_abs = inv[mask] * initial_inventory
            missing = initial_inventory - inv_abs[:, -1]
            avg_missing = np.mean(missing)
            std_missing = np.std(missing)
            max_missing = np.max(missing)
            print(f'[{lbl}] {pct:.4f}% of runs did not reach full inventory')
            print(f'    Missing volume: avg={avg_missing:.2f}, std={std_missing:.2f}, max={max_missing:.2f}\n')
        else:
            print(f'[{lbl}] 0.0000% of runs did not reach full inventory')

        mu = inv.mean(axis=0)
        s = inv.std(axis=0)
        x_vals = np.arange(1, len(trader_times) + 1)
        if lbl != 'DDQN':
            line, = ax2.plot(x_vals, mu, label=lbl, color=non_red_colors[kk])
            ax2.fill_between(x_vals, mu - s/2, mu + s/2, alpha=0.2, color=non_red_colors[kk])
        else:
            line, = ax2.plot(x_vals, mu, linestyle='--', color='C3', label=lbl)
            ax2.fill_between(x_vals, mu - s/2, mu + s/2, alpha=0.2, color='C3')

    ax2.set_xlim(1, len(trader_times))
    ax2.set_xlabel('Trader Step')
    ax2.set_ylabel('Executed Inventory (%)')
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()

    # ===============================
    # Figure 3: DDQN Episode Lengths (PDF + ECDF)
    # ===============================
    times = np.array(ddqn_lengths)
    fig3, ax3 = plt.subplots(figsize=(4,3))
    max_len = len(trader_times)
    bin_edges = np.arange(0.5, max_len + 1.5, 1)

    # left axis: histogram
    ax3.hist(times, bins=bin_edges, density=True, alpha=0.3, color='C0', label='Density')

    # right axis: ECDF
    ax3b = ax3.twinx()
    sorted_t = np.sort(times)
    ecdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    ax3b.step(sorted_t, ecdf, where='post', color='k', linewidth=0.5, alpha=0.8, label='ECDF')

    # labels
    ax3.set_xlabel('Trader Step')
    ax3.set_ylabel('Density', color='k')
    ax3b.set_ylabel('ECDF', color='k')
    ax3.set_xlim(0.5, max_len + 0.5)
    ax3b.set_xlim(ax3.get_xlim())
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax3.yaxis.grid(True, which='major', alpha=0.4) 
    ax3.xaxis.grid(False)                         
    ax3b.grid(False)                            

    # twin-axes
    ax3.set_zorder(2)
    ax3.patch.set_visible(False)
    ax3b.set_zorder(1)
    ax3b.patch.set_visible(False)
    ax3b.set_ylim(0.0, 1.0)
    ax3b.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))

    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc='best')
    fig3.tight_layout()

    plt.show()


