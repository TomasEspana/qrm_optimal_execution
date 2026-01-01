import numpy as np 
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import pandas as pd
from scipy import stats
import re



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



######################################
######################################
######################################



def plotly_eye_to_view_init(eye):
    """
    Convert a Plotly camera 'eye' dict to Matplotlib view_init(elev, azim).
    eye: dict like {'x': ..., 'y': ..., 'z': ...}
    """
    x, y, z = eye["x"], eye["y"], eye["z"]
    azim = np.degrees(np.arctan2(y, x))
    r_xy = np.hypot(x, y)
    elev = np.degrees(np.arctan2(z, r_xy))
    return elev, azim


def save_action_surface_pdf(
    elev, azim, action_idx, q_values, n_grid, time_pct, inv_pct, filename="q_surface_action.pdf",
    global_vmin=None, global_vmax=None, cmap="plasma"
):
    Z = q_values[:, action_idx].reshape(n_grid, n_grid)

    vmin = np.nanmin(Z) if global_vmin is None else global_vmin
    vmax = np.nanmax(Z) if global_vmax is None else global_vmax

    # Bigger canvas; we’ll leave generous margins
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # cleaner text (less perspective distortion)

    # Surface colored by Q-value (viridis ~ Plotly)
    surf = ax.plot_surface(
        time_pct, inv_pct, Z,
        cmap=cmap, vmin=vmin, vmax=vmax,
        rstride=1, cstride=1, linewidth=0, antialiased=True
    )

    # Labels & ticks
    ax.set_xlabel("Time (%)", labelpad=10, fontsize=15)
    ax.set_ylabel("Remaining Inventory (%)", labelpad=10, fontsize=15)

    ax.zaxis.set_rotate_label(False)                 # keep our rotation
    ax.set_zlabel("Q-value", rotation=90, labelpad=4, fontsize=15)  # bottom→top

    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(pad=4)
    ax.set_box_aspect((1, 1, 0.8))
    ax.view_init(elev=elev, azim=azim)

    # Slim colorbar with clear label
    cbar = fig.colorbar(surf, ax=ax, shrink=0.82, pad=0.10)
    # cbar.set_label("Q-value")

    # Give extra room so axis labels never clip
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.16, top=0.98)

    # Safe vector export
    fig.savefig(filename, format="pdf", bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)




def compute_t_test(types, run_names, size=20_000, reference='ddqn', side='greater'):
    dic_is = {}
    for typ, run_name in zip(types, run_names):
        with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/{run_name}.pkl', 'rb') as f:
            dic = pickle.load(f)
        final_is = np.array(dic['final_is'])
        final_is = np.random.choice(final_is, size=size, replace=False)
        dic_is[typ] = final_is

    for typ in types:
        if typ != reference:
            t_stat, p_value = stats.ttest_ind(dic_is[reference], dic_is[typ], equal_var=False, alternative=side)
            print(f'T-test between {reference} and {typ}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}')
            if p_value < 0.05:
                print(f'Statistically significant difference between {reference} and {typ} (p < 0.05)')
            print('\n')





def plot_mean_distance_boxplots(executed, trader_times, run_id=None):
    """
    For each run in `executed` (a dict of run_id → list of executed volumes),
    compute the mean number of zeros between consecutive non-zero volumes
    (i.e. the “gap” length), group those means by the length of the run,
    and draw a boxplot for each possible run length 1..len(trader_times)-1.
    
    Parameters
    ----------
    executed : dict[int, list[int]]
        Keys are run IDs; values are lists of volumes (integers) executed
        at each trader time step (zeros allowed).
    trader_times : Sequence
        Any sequence whose length defines the maximum possible run length.
    """
    # x-axis values: 1, 2, ..., len(trader_times)-1
    x_vals = list(range(1, len(trader_times)))
    
    def mean_zero_gap(vols):
        # find indices of non-zero executions
        idxs = [i for i, v in enumerate(vols) if v != 0]
        # need at least two to form a “gap”
        if len(idxs) < 2:
            return None
        # gap = count of zeros between each pair of non-zeros
        gaps = [idxs[i+1] - idxs[i] - 1 for i in range(len(idxs)-1)]
        return sum(gaps) / len(gaps)
    
    # initialize storage for each possible length
    grouped = {L: [] for L in x_vals}
    
    # compute mean gap for each run, bucket by run length
    for i, vols in executed.items():
        L = len(vols)
        if L in grouped:
            mg = mean_zero_gap(vols) 
            if mg is not None:
                grouped[L].append(mg)
    
    # prepare data & positions for boxplot (skip lengths with no data)
    data   = []
    pos    = []
    for L in x_vals:
        if grouped[L]:
            data.append(grouped[L])
            pos.append(L)
    
    # draw
    plt.figure(figsize=(5, 4))
    plt.boxplot(data, positions=pos, widths=0.6, showfliers=False)
    plt.xlabel('Episode length', fontsize=13)
    plt.ylabel('Gap Mean', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.tight_layout()
    plt.savefig(f"/scratch/network/te6653/qrm_optimal_execution/plots/gaps/{run_id}.pdf", bbox_inches="tight")

    plt.figure(figsize=(5, 4))
    plt.boxplot(data, positions=pos, widths=0.6)
    plt.xlabel('Episode length', fontsize=13)
    plt.ylabel('Gap Mean', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.tight_layout()
    plt.savefig(f"/scratch/network/te6653/qrm_optimal_execution/plots/gaps/{run_id}_outliers.pdf", bbox_inches="tight")
    plt.show()





def plot_intra_mean_distance_boxplots(executed, trader_times, run_id=None):
    """
    For each run in `executed` (a dict of run_id → list of executed volumes),
    compute the mean number of zeros between consecutive non-zero volumes
    (i.e. the “gap” length), group those means by the length of the run,
    and draw a boxplot for each possible run length 1..len(trader_times)-1.
    
    Parameters
    ----------
    executed : dict[int, list[int]]
        Keys are run IDs; values are lists of volumes (integers) executed
        at each trader time step (zeros allowed).
    trader_times : Sequence
        Any sequence whose length defines the maximum possible run length.
    """
    # x-axis values: 1, 2, ..., len(trader_times)-1
    x_vals = list(range(1, len(trader_times)))
    
    def var_zero_gap(vols):
        # find indices of non-zero executions
        idxs = [i for i, v in enumerate(vols) if v != 0]
        # need at least two to form a “gap”
        if len(idxs) < 2:
            return None
        # gap = count of zeros between each pair of non-zeros
        gaps = [idxs[i+1] - idxs[i] - 1 for i in range(len(idxs)-1)]
        return np.var(gaps)
    
    # initialize storage for each possible length
    grouped = {L: [] for L in x_vals}
    
    # compute mean gap for each run, bucket by run length
    for i, vols in executed.items():
        L = len(vols)
        if L in grouped:
            mg = var_zero_gap(vols) 
            if mg is not None:
                grouped[L].append(mg)
    
    # prepare data & positions for boxplot (skip lengths with no data)
    data   = []
    pos    = []
    for L in x_vals:
        if grouped[L]:
            data.append(grouped[L])
            pos.append(L)
    
    # draw
    plt.figure(figsize=(5, 4))
    plt.boxplot(data, positions=pos, widths=0.6, showfliers=False)
    plt.xlabel('Episode length', fontsize=13)
    plt.ylabel('Gap Variance', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.tight_layout()
    plt.savefig(f"/scratch/network/te6653/qrm_optimal_execution/plots/gaps/variance_{run_id}.pdf", bbox_inches="tight")

    plt.figure(figsize=(5, 4))
    plt.boxplot(data, positions=pos, widths=0.6)
    plt.xlabel('Episode length', fontsize=13)
    plt.ylabel('Gap Variance', fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.set_xlim(min(pos) - 0.5, max(pos) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.tight_layout()
    plt.savefig(f"/scratch/network/te6653/qrm_optimal_execution/plots/gaps/variance_{run_id}_outliers.pdf", bbox_inches="tight")
    plt.show()



def plot_overview(
    run_names,
    run_labels,
    initial_inventory,
    trader_times,
    title,
    metric='executed',
    kde_only=True,
    best_labels=None,
    run_id=None, 
    save_fig=True
):
    non_red_colors = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    final_is = {}
    actions = {}
    ddqn_lengths = []

    for lbl, rn in zip(run_labels, run_names):
        with open(f'/scratch/network/te6653/qrm_optimal_execution/data_wandb/dictionaries/{rn}.pkl', 'rb') as f:
            dic = pickle.load(f)
        final_is[lbl] = np.asarray(dic['final_is'])
        exec_dict = dic[metric]
        actions[lbl] = exec_dict
        if lbl == 'DDQN':
            seqs = [exec_dict[i] for i in sorted(exec_dict)]
            ddqn_lengths = [len(seq) for seq in seqs]

    # --- Prepare x-grids for IS plots ---
    maxi = max(np.max(np.abs(v)) for v in final_is.values())
    x_sym = np.linspace(-maxi, maxi, 1000)
    all_vals = np.concatenate(list(final_is.values()))
    x_auto = np.linspace(all_vals.min(), all_vals.max(), 1000)

    # filename prefix
    safe_title = re.sub(r'[^-\w]+', '_', title) if title else 'overview'
    prefix = f"{safe_title}" + (f"_{run_id}" if run_id is not None else "")

    # ===============================
    # Figure 1: IS KDE (all methods), symmetric limits
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

    # ax1.set_title(f"{title} — IS (all methods, symmetric)", fontsize=11)
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Density')
    extra = 1 # 3 for dim 4
    ax1.set_xlim(x_sym.min() + extra, x_sym.max() - extra)
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    if save_fig:
        fig1.savefig(f"/scratch/network/te6653/qrm_optimal_execution/plots/overview/{run_id}_subplot_1.pdf", bbox_inches="tight")


    # ===============================
    # Figure 2: IS KDE (best performing) auto-limits
    # ===============================
    fig2, ax2 = plt.subplots(figsize=(4,3))
    labels_to_plot = best_labels if (best_labels is not None) else final_is.keys()

    if not kde_only:
        for lbl in labels_to_plot:
            vals = final_is[lbl]
            ax2.hist(vals, bins=27, density=True, alpha=0.3, label=f'{lbl} hist')

    for lbl in labels_to_plot:
        vals = final_is[lbl]
        kde = gaussian_kde(vals)
        y = kde(x_auto)
        if lbl != 'DDQN':
            ax2.plot(x_auto, y, label=lbl)
        else:
            ax2.plot(x_auto, y, linestyle='--', color='C3', label=lbl)

    # ax2.set_title(f"{title} — IS (best performing, auto-limits)", fontsize=11)
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Density')
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    if save_fig:
        fig2.savefig(f"/scratch/network/te6653/qrm_optimal_execution/plots/overview/{run_id}_subplot_2.pdf", bbox_inches="tight")

    # ===============================
    # Figure 3: Avg. Inventory Trajectory (μ ± σ/2)
    # ===============================
    fig3, ax3 = plt.subplots(figsize=(4,3))
    for kk, (lbl, exec_dict) in enumerate(actions.items()):
        seqs = [exec_dict[i] for i in sorted(exec_dict)]
        inv = np.zeros((len(seqs), len(trader_times)))
        for i, seq in enumerate(seqs):
            inv[i, :len(seq)] = seq
        inv = np.cumsum(inv, axis=1) / initial_inventory

        # Stats
        mask = inv[:, -1] < 1 # Runs that did not reach full inventory
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
            line, = ax3.plot(x_vals, mu, label=lbl, color=non_red_colors[kk])
            ax3.fill_between(x_vals, mu - s/2, mu + s/2, alpha=0.2, color=non_red_colors[kk])
        else:
            line, = ax3.plot(x_vals, mu, linestyle='--', color='C3', label=lbl)
            ax3.fill_between(x_vals, mu - s/2, mu + s/2, alpha=0.2, color='C3')

    ax3.set_xlim(1, len(trader_times))
    ax3.set_xlabel('Trader Step')
    ax3.set_ylabel('Executed Inventory (%)')
    # ax3.set_title(f"{title} — Avg. Inventory Trajectory (μ ± σ/2)", fontsize=11)
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()
    if save_fig:
        fig3.savefig(f"/scratch/network/te6653/qrm_optimal_execution/plots/overview/{run_id}_subplot_3.pdf", bbox_inches="tight")

    # ===============================
    # Figure 4: DDQN Episode Lengths (PDF + ECDF)
    # ===============================
    # --- Single plot (DDQN only) ---
    times = np.array(ddqn_lengths)
    fig4, ax4 = plt.subplots(figsize=(4,3))
    max_len = len(trader_times)
    bin_edges = np.arange(0.5, max_len + 1.5, 1)

    # Left axis: histogram (Density)
    ax4.hist(times, bins=bin_edges, density=True, alpha=0.3, color='C0', label='Density')

    # Right axis: ECDF
    ax4b = ax4.twinx()
    sorted_t = np.sort(times)
    ecdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    ax4b.step(sorted_t, ecdf, where='post', color='k', linewidth=0.5, alpha=0.8, label='ECDF')

    # Labels
    ax4.set_xlabel('Trader Step')
    ax4.set_ylabel('Density', color='k')
    ax4b.set_ylabel('ECDF', color='k')
    # ax4.set_title(f"{title} — Episode Length", fontsize=11)

    # X limits shared
    ax4.set_xlim(0.5, max_len + 0.5)
    ax4b.set_xlim(ax4.get_xlim())

    # X ticks as integers
    ax4.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune='both'))
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # ---- GRID FIX (match final code) ----
    ax4.yaxis.grid(True, which='major', alpha=0.4)   # only left-axis horizontal grid
    ax4.xaxis.grid(False)                            # no vertical grid
    ax4b.grid(False)                                 # right axis draws no grid

    # Twin-axes layering cosmetics
    ax4.set_zorder(2)
    ax4.patch.set_visible(False)
    ax4b.set_zorder(1)
    ax4b.patch.set_visible(False)

    # ECDF ticks and limits
    ax4b.set_ylim(0.0, 1.0)
    ax4b.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))

    # Single combined legend
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, loc='best')

    fig4.tight_layout()
    if save_fig:
        fig4.savefig(f"/scratch/network/te6653/qrm_optimal_execution/plots/overview/{run_id}_subplot_4.pdf", bbox_inches="tight")
    plt.show()


