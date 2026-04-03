#### READ ME ####
# per condition 1per agarose, 2per agarose, 3per agarose we have one dataset of whole brain activity of one worm.
# higher agarose on top the worm crawls means worm make more effort to move, and crawl more slowly.
# but what happens in the brain when its hard to move ???
# Per condition you have one dataset of whole brain activity of one worm, and you have the behavior annotation of this worm, so you can see when it reverses and when it goes forward.
# the files you have are:
# 1. .png file that shows the full activity heatmap, every row is a neuron, every column is a time point, and the color is the activity of the neuron at that time point. The neurons are sorted by their activity pattern, so neurons with similar activity patterns are close to each other in the heatmap.
# 2. ..ratio.h5 file that contains the traces of all the neurons, in a dataframe
# 3. ..neuron_ids.xlsx file that contains the ID of the neurons, so you can know which neuron is which, and what is its name.
# 4. ..beh_annotation.csv file that contains the automatic behavior annotation of the worm, so you can know when it is reversing and when it is going forward.
# 5. ..manual_reversal_events that contains manually defined reversal events.
# This tutorial will guide you thorugh opening these files, and analyzing the data, and trying to understand what is going on in the brain of the worm when it is hard to move.

### ---- ####

# - imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from datetime import datetime
import os

from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# All 3 agarose conditions with their file paths
conditions = {
    '1per': {
        'traces':      r'.\1per\2024-07-12_15-57_1per_worm2-2024-07-12_ratio.h5',
        'neuron_ids':  r'.\1per\2024-07-12_15-57_1per_worm2-2024-07-12_neuron_ids.xlsx',
        'beh_annotation': r'.\1per\2024-07-12_15-57_1per_worm2-2024-07-12_beh_annotation.csv',
    },
    '2per': {
        'traces':      r'.\2per\2024-07-18_14-31_2per_worm1-2024-07-18_ratio.h5',
        'neuron_ids':  r'.\2per\2024-07-18_14-31_2per_worm1-2024-07-18_neuron_IDs.xlsx',
        'beh_annotation': r'.\2per\2024-07-18_14-31_2per_worm1-2024-07-18_beh_annotation.csv',
    },
    '3per': {
        'traces':      r'.\3per\2023-05-11_16-11_ZIM2165_NP_3per_worm2_fixed-2023-05-11_ratio.h5',
        'neuron_ids':  r'.\3per\2023-05-11_16-11_ZIM2165_NP_3per_worm2_fixed-2023-05-11_neuron_IDs.xlsx',
        'beh_annotation': r'.\3per\2023-05-11_16-11_ZIM2165_NP_3per_worm2_fixed-2023-05-11_beh_annotation.csv',
    },
}

# Setup a PDF so that we don't spam 16 pop-up windows when running
pdf_path = 'dataset_analysis_plots.pdf'
pdf_writer = PdfPages(pdf_path)

save_dir = 'saved_figures'
os.makedirs(save_dir, exist_ok=True)

# Collect PC1 loadings per neuron name across conditions (only for identified neurons)
pc1_signs_across_conditions = {}  # {neuron_name: {condition: loading_value}}

# Collect per-condition clustering data for Excel export
all_cluster_rows = []

# Collect per-condition dendrogram group data for Excel export
all_dendro_group_rows = []

# Collect behavior correlation data for Excel export
all_corr_rows = []

for condition_name, paths in conditions.items():

    # ── Load behavioral annotations ──────────────────────────────────────────
    beh_df = pd.read_csv(paths['beh_annotation'], header=None)
    beh_df = beh_df.iloc[::24]
    beh_df.index = beh_df.index // 24

    n_timepoints = len(pd.read_hdf(paths['traces']))  # total frames in trace
    reverse_time_points = [t for t in map(int, beh_df[beh_df[1] == 1].index)  if t < n_timepoints]
    forward_time_points = [t for t in map(int, beh_df[beh_df[1] == -1].index) if t < n_timepoints]
    stay_time_points    = [t for t in map(int, beh_df[beh_df[1] == 0].index)  if t < n_timepoints]


    # ── Load neuron IDs ───────────────────────────────────────────────────────
    cond_neuron_id_df = pd.read_excel(paths['neuron_ids'])

    # ── Load & interpolate traces ─────────────────────────────────────────────
    cond_trace_df = pd.read_hdf(paths['traces'])
    cond_trace_df = cond_trace_df.interpolate(method='linear', axis=0, limit_direction='both')

    # ── Cluster neurons by similarity & save dendrogram ──────────────────────
    cond_distance_array = squareform(pairwise_distances(cond_trace_df.T, metric='correlation'))
    cond_linkage_matrix = linkage(cond_distance_array, method='average', metric='correlation')

    # Cut dendrogram into 3 groups and assign colors
    dendro_groups = fcluster(cond_linkage_matrix, t=3, criterion='maxclust')  # 1-indexed
    dendro_group_colors = {1: 'steelblue', 2: 'tomato', 3: 'seagreen'}

    # Build a mapping from leaf index -> group color for dendrogram coloring
    def get_link_color(link_id):
        """Color links based on whether all descendant leaves belong to the same group."""
        n_samples = len(cond_linkage_matrix) + 1
        if link_id < n_samples:
            return dendro_group_colors[dendro_groups[link_id]]
        # internal node: check if all leaves below share one group
        row = cond_linkage_matrix[link_id - n_samples]
        left, right = int(row[0]), int(row[1])
        def get_leaves(idx):
            if idx < n_samples:
                return [idx]
            r = cond_linkage_matrix[idx - n_samples]
            return get_leaves(int(r[0])) + get_leaves(int(r[1]))
        all_leaves = get_leaves(left) + get_leaves(right)
        groups_here = set(dendro_groups[l] for l in all_leaves)
        if len(groups_here) == 1:
            return dendro_group_colors[groups_here.pop()]
        return 'gray'

    dend_fig, dend_ax = plt.subplots(figsize=(15, 5))
    cond_dendrogram = dendrogram(
        cond_linkage_matrix, no_labels=True, ax=dend_ax,
        link_color_func=get_link_color
    )
    # Add legend for dendrogram groups
    from matplotlib.patches import Patch
    dend_legend = [Patch(facecolor=dendro_group_colors[i], label=f'Group {i}') for i in sorted(dendro_group_colors)]
    dend_ax.legend(handles=dend_legend, loc='upper right', fontsize=8)
    dend_ax.set_title(f'Dendrogram - {condition_name} agarose')
    dend_fig.tight_layout()
    cond_trace_df = cond_trace_df.iloc[:, cond_dendrogram['leaves']]

    # ── Dendrogram Heatmap ────────────────────────────────────────────────────
    dend_heatmap_fig, dend_heatmap_ax = plt.subplots(figsize=(10, 5))
    im = dend_heatmap_ax.imshow(cond_trace_df.T, cmap='viridis', aspect='auto')
    dend_heatmap_fig.colorbar(im, ax=dend_heatmap_ax, label='Activity')
    dend_heatmap_ax.set_xlabel('Time')
    dend_heatmap_ax.set_ylabel('Neurons')
    dend_heatmap_ax.set_title(f'Heatmap - {condition_name} agarose')
    dend_heatmap_fig.tight_layout()

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_cond   = PCA(n_components=10)
    pca_result = pca_cond.fit_transform(cond_trace_df)

    # ── 3D PCA line plot (Task 1) ─────────────────────────────────────────────
    fig_3d = plt.figure()
    ax_3d  = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(pca_result[forward_time_points, 0], pca_result[forward_time_points, 1], pca_result[forward_time_points, 2], color='blue',  label='forward')
    ax_3d.plot(pca_result[reverse_time_points, 0], pca_result[reverse_time_points, 1], pca_result[reverse_time_points, 2], color='red',   label='reverse')
    ax_3d.plot(pca_result[stay_time_points,    0], pca_result[stay_time_points,    1], pca_result[stay_time_points,    2], color='green', label='stay')
    ax_3d.set_xlabel('PC1')
    ax_3d.set_ylabel('PC2')
    ax_3d.set_zlabel('PC3')
    ax_3d.set_title(f'3D PCA - {condition_name} agarose')
    ax_3d.legend()

    # ── 2D PCA scatter plot (PC1 vs PC2) ──────────────────────────────────────
    fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
    ax_2d.scatter(pca_result[forward_time_points, 0], pca_result[forward_time_points, 1], color='blue', label='forward', alpha=0.6, s=15)
    ax_2d.scatter(pca_result[reverse_time_points, 0], pca_result[reverse_time_points, 1], color='red',  label='reverse', alpha=0.6, s=15)
    ax_2d.scatter(pca_result[stay_time_points, 0],    pca_result[stay_time_points, 1],    color='green', label='stay', alpha=0.6, s=15)
    ax_2d.set_xlabel('PC1')
    ax_2d.set_ylabel('PC2')
    ax_2d.set_title(f'2D PCA (PC1 vs PC2) - {condition_name} agarose')
    ax_2d.legend()

    # ── Scree Plot for Normal PCA ─────────────────────────────────────────────
    scree_norm_fig, scree_norm_ax = plt.subplots(figsize=(6, 4))
    n_comp_norm = len(pca_cond.explained_variance_ratio_)
    scree_norm_ax.plot(range(1, n_comp_norm + 1), pca_cond.explained_variance_ratio_ * 100, marker='o', linestyle='--')
    scree_norm_ax.set_title(f'Scree Plot (Normal PCA) - {condition_name} agarose')
    scree_norm_ax.set_xlabel('Principal Component')
    scree_norm_ax.set_ylabel('Explained Variance (%)')
    scree_norm_ax.set_xticks(range(1, n_comp_norm + 1))
    scree_norm_fig.tight_layout()

    # ── PCA on Transposed Traces (Neurons as Samples) ─────────────────────────
    scaler = StandardScaler()
    std_trace_matrix = scaler.fit_transform(cond_trace_df)
    std_trace_df_T = pd.DataFrame(std_trace_matrix, columns=cond_trace_df.columns, index=cond_trace_df.index).T
    
    pca_transposed = PCA(n_components=10)
    pca_t_result = pca_transposed.fit_transform(std_trace_df_T)

    scree_fig, scree_ax = plt.subplots(figsize=(6, 4))
    n_comp_t = len(pca_transposed.explained_variance_ratio_)
    scree_ax.plot(range(1, n_comp_t + 1), pca_transposed.explained_variance_ratio_ * 100, marker='o', linestyle='--')
    scree_ax.set_title(f'Scree Plot (Transposed PCA) - {condition_name} agarose')
    scree_ax.set_xlabel('Principal Component')
    scree_ax.set_ylabel('Explained Variance (%)')
    scree_ax.set_xticks(range(1, n_comp_t + 1))
    scree_fig.tight_layout()



    # ── Task 2: PC1 activity ───────────────────────────────────────────────────
    bx_fig, bx = plt.subplots()
    bx.plot(pca_result[:, 0], label='PC1', color='blue')
    bx.plot(pca_result[:, 1], label='PC2', color='orange')
    aval_col = cond_neuron_id_df.loc[cond_neuron_id_df['ID1'] == 'AVAL', 'Neuron ID']
    avbl_col = cond_neuron_id_df.loc[cond_neuron_id_df['ID1'] == 'AVBL', 'Neuron ID'] # Checking AVBL because i saw it has a strong negative loading for pc1
    nsml_col = cond_neuron_id_df.loc[cond_neuron_id_df['ID1'] == 'NSML', 'Neuron ID'] # Checking NSML because i saw it has zero loading for pc1

    if not aval_col.empty:
         bx.plot(cond_trace_df[aval_col], color='black', label='AVAL')
    if not avbl_col.empty:
        bx.plot(cond_trace_df[avbl_col], color='green',  label='AVBL')
    if not nsml_col.empty:
        bx.plot(cond_trace_df[nsml_col], color='red',  label='NSML')

    # Overlay behavior annotation on a secondary y-axis
    # beh_df is already downsampled by 24 to match the trace timescale
    beh_ax = bx.twinx()
    beh_ax.step(beh_df.index, beh_df[1], color='red', alpha=0.4, linewidth=1, label='Behavior')
    beh_ax.set_ylabel('Behavior (1=reverse, 0=stay, -1=forward)', color='red', fontsize=8)
    beh_ax.tick_params(axis='y', labelcolor='red')
    beh_ax.set_yticks([1, 0, -1])
    beh_ax.set_yticklabels(['reverse', 'stay', 'forward'], fontsize=7)
    beh_ax.set_ylim(-2, 2)  # give some padding so the step line doesn't overlap the traces

    bx.set_title(f'PC1 vs neurons activity - {condition_name} agarose')
    # Combine legends from both axes
    lines1, labels1 = bx.get_legend_handles_labels()
    lines2, labels2 = beh_ax.get_legend_handles_labels()
    bx.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)


    # ── Task 3: PC1 contribution bar chart ────────────────────────────────────
    cond_components_pc1 = pca_cond.components_[0].tolist()

    # Map neuron IDs to names (fall back to column name if no ID1)
    cond_id_mapping   = dict(zip(cond_neuron_id_df['Neuron ID'], cond_neuron_id_df['ID1']))
    cond_neuron_names = [
        cond_id_mapping.get(col, col) if pd.notna(cond_id_mapping.get(col, col)) else col
        for col in cond_trace_df.columns
    ]

    # ── KMeans clustering of PC1 loadings (3 clusters) ─────────────────────────
    loadings_array = np.array(cond_components_pc1).reshape(-1, 1)

    # ── KMeans Elbow graph for PC1 loadings ──────────────────────────────────
    inertias = []
    K_range = range(1, min(10, len(loadings_array)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(loadings_array)
        inertias.append(km.inertia_)
    
    elbow_fig, elbow_ax = plt.subplots(figsize=(6, 4))
    elbow_ax.plot(K_range, inertias, marker='o', linestyle='--')
    elbow_ax.set_title(f'K-Means Elbow Graph (PC1 Loadings) - {condition_name} agarose')
    elbow_ax.set_xlabel('Number of clusters (k)')
    elbow_ax.set_ylabel('Inertia')
    elbow_ax.set_xticks(list(K_range))
    elbow_fig.tight_layout()

    # Cluster 0 = most positive, Cluster 1 = middle, Cluster 2 = most negative
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=100)
    raw_labels = kmeans.fit_predict(loadings_array)
    # Sort cluster IDs by their center values (descending: biggest positive → 0)
    center_order = np.argsort(kmeans.cluster_centers_[:, 0])[::-1]  # indices sorted high→low
    remap = {old: new for new, old in enumerate(center_order)}
    cluster_labels = np.array([remap[l] for l in raw_labels])
    cluster_colors_map = {0: 'steelblue', 1: 'tomato', 2: 'seagreen'}

    # ── Scatter Plot of Transposed PCA (Colored by PC1 KMeans) ────────────────
    scatter_t_fig, scatter_t_ax = plt.subplots(figsize=(8, 6))
    scatter_colors = [cluster_colors_map[c] for c in cluster_labels]
    scatter_t_ax.scatter(pca_t_result[:, 0], pca_t_result[:, 1], alpha=0.7, color=scatter_colors, s=20)
    scatter_t_ax.set_title(f'Neurons in PC Space (Transposed PCA) - {condition_name} agarose')
    scatter_t_ax.set_xlabel(f'PC1 ({pca_transposed.explained_variance_ratio_[0]*100:.1f}%)')
    scatter_t_ax.set_ylabel(f'PC2 ({pca_transposed.explained_variance_ratio_[1]*100:.1f}%)')
    scatter_legend = [Patch(facecolor=cluster_colors_map[i], label=f'Cluster {i}') for i in sorted(cluster_colors_map)]
    scatter_t_ax.legend(handles=scatter_legend, loc='best', fontsize=8)
    scatter_t_fig.tight_layout()

    # ── Scatter Plot of Transposed PCA (Colored by PC2 loading) ──────────────
    scatter_t2_fig, scatter_t2_ax = plt.subplots(figsize=(8, 6))
    cond_components_pc2 = pca_cond.components_[1].tolist()
    sc = scatter_t2_ax.scatter(pca_t_result[:, 0], pca_t_result[:, 1], alpha=0.7, c=cond_components_pc2, cmap='coolwarm', s=20)
    scatter_t2_fig.colorbar(sc, ax=scatter_t2_ax, label='PC2 Loading')
    scatter_t2_ax.set_title(f'Neurons in PC Space (Colored by PC2 loading) - {condition_name} agarose')
    scatter_t2_ax.set_xlabel(f'PC1 ({pca_transposed.explained_variance_ratio_[0]*100:.1f}%)')
    scatter_t2_ax.set_ylabel(f'PC2 ({pca_transposed.explained_variance_ratio_[1]*100:.1f}%)')
    scatter_t2_fig.tight_layout()

    # Store cluster assignments for Excel export
    for name, loading, cluster in zip(cond_neuron_names, cond_components_pc1, cluster_labels):
        all_cluster_rows.append({
            'Neuron':     name,
            'Condition':  condition_name,
            'PC1 loading': round(loading, 4),
            'Cluster':    int(cluster),
        })

    # ── Correlation with Behavior ─────────────────────────────────────────────
    min_len = min(len(beh_df), len(cond_trace_df))
    beh_signal = beh_df[1].iloc[:min_len].values.astype(float)
    
    cond_correlations = []
    
    for name, col, cluster in zip(cond_neuron_names, cond_trace_df.columns, cluster_labels):
        trace_sig = cond_trace_df[col].iloc[:min_len].values.astype(float)
        valid = ~(np.isnan(trace_sig) | np.isnan(beh_signal))
        if not np.any(valid) or np.std(trace_sig[valid]) == 0 or np.std(beh_signal[valid]) == 0:
            corr = np.nan
        else:
            corr, _ = pearsonr(trace_sig[valid], beh_signal[valid])
            
        cond_correlations.append(corr)
        
        all_corr_rows.append({
            'Neuron': name,
            'Condition': condition_name,
            'Correlation to Behavior': round(corr, 4) if not np.isnan(corr) else np.nan,
            'PC1 Cluster Color': cluster_colors_map[cluster]
        })

    # ── Sorted Bar Chart (by PC1 loading, colored by cluster) ─────────────────
    cond_sorted_pairs = sorted(zip(cond_neuron_names, cond_components_pc1, cluster_labels), key=lambda x: x[1])
    cond_sorted_names, cond_sorted_components, cond_sorted_clusters = zip(*cond_sorted_pairs)
    sorted_colors = [cluster_colors_map[c] for c in cond_sorted_clusters]

    bar_fig, bar_ax = plt.subplots(figsize=(15, 6))
    bar_ax.bar(cond_sorted_names, cond_sorted_components, color=sorted_colors)
    bar_ax.set_title(f'PC1 Contribution (K-Means colored) - {condition_name} agarose', fontsize=14)
    bar_ax.set_xlabel('Neuron', fontsize=12)
    bar_ax.set_ylabel('Contribution to PC1', fontsize=12)
    bar_ax.tick_params(axis='x', rotation=90, labelsize=8)
    # Add cluster legend
    legend_patches = [Patch(facecolor=cluster_colors_map[i], label=f'Cluster {i}') for i in sorted(cluster_colors_map)]
    bar_ax.legend(handles=legend_patches, loc='upper left', fontsize=8)
    bar_fig.tight_layout()

    # ── Sorted Bar Chart (by PC2 loading, colored by PC1 cluster) ────────────
    cond_sorted_pairs_pc2 = sorted(zip(cond_neuron_names, cond_components_pc2, cluster_labels), key=lambda x: x[1])
    pc2_sorted_names, pc2_sorted_components, pc2_sorted_clusters = zip(*cond_sorted_pairs_pc2)
    pc2_sorted_colors = [cluster_colors_map[c] for c in pc2_sorted_clusters]

    pc2_bar_fig, pc2_bar_ax = plt.subplots(figsize=(15, 6))
    pc2_bar_ax.bar(pc2_sorted_names, pc2_sorted_components, color=pc2_sorted_colors)
    pc2_bar_ax.set_title(f'PC2 Contribution (colored by PC1 K-Means) - {condition_name} agarose', fontsize=14)
    pc2_bar_ax.set_xlabel('Neuron', fontsize=12)
    pc2_bar_ax.set_ylabel('Contribution to PC2', fontsize=12)
    pc2_bar_ax.tick_params(axis='x', rotation=90, labelsize=8)
    pc2_bar_ax.legend(handles=legend_patches, loc='upper left', fontsize=8)
    pc2_bar_fig.tight_layout()

    # ── PC1 Loadings Heatmap (neurons ordered by sorted PC1 loadings) ────────
    # Reorder trace columns by PC1 loading (same order as the sorted bar chart)
    pc1_sorted_col_indices = [list(cond_trace_df.columns).index(
        cond_neuron_id_df.loc[cond_neuron_id_df['ID1'] == n, 'Neuron ID'].values[0]
        if n in cond_neuron_id_df['ID1'].values
        else n
    ) for n in cond_sorted_names]
    # Simpler approach: sort columns directly by their PC1 loading
    col_loading_pairs = list(zip(cond_trace_df.columns, cond_components_pc1))
    col_loading_pairs.sort(key=lambda x: x[1])  # sort by loading ascending
    sorted_cols = [c for c, _ in col_loading_pairs]
    pc1_sorted_trace = cond_trace_df[sorted_cols]

    pc1_loadings_heatmap_fig, pc1_loadings_heatmap_ax = plt.subplots(figsize=(10, 5))
    im2 = pc1_loadings_heatmap_ax.imshow(pc1_sorted_trace.T, cmap='viridis', aspect='auto')
    pc1_loadings_heatmap_fig.colorbar(im2, ax=pc1_loadings_heatmap_ax, label='Activity')
    pc1_loadings_heatmap_ax.set_xlabel('Time')
    pc1_loadings_heatmap_ax.set_ylabel('Neurons (sorted by PC1 loading)')
    pc1_loadings_heatmap_ax.set_title(f'Heatmap (PC1 sorted) + Behavior - {condition_name} agarose')

    # Overlay behavior annotation on a secondary y-axis
    hm_beh_ax = pc1_loadings_heatmap_ax.twinx()
    hm_beh_ax.step(beh_df.index, beh_df[1], color='red', alpha=0.5, linewidth=2, label='Behavior')
    hm_beh_ax.set_ylabel('Behavior', color='red', fontsize=8)
    hm_beh_ax.tick_params(axis='y', labelcolor='red')
    hm_beh_ax.set_yticks([1, 0, -1])
    hm_beh_ax.set_yticklabels(['reverse', 'stay', 'forward'], fontsize=7)
    hm_beh_ax.set_ylim(-2, 2)  # Give some padding to position the step line cleanly
    
    pc1_loadings_heatmap_fig.tight_layout()

    # ── PC2 Loadings Heatmap (neurons ordered by sorted PC2 loadings) ────────
    col_loading_pairs_pc2 = list(zip(cond_trace_df.columns, cond_components_pc2))
    col_loading_pairs_pc2.sort(key=lambda x: x[1])  # sort by loading ascending
    sorted_cols_pc2 = [c for c, _ in col_loading_pairs_pc2]
    pc2_sorted_trace = cond_trace_df[sorted_cols_pc2]

    pc2_loadings_heatmap_fig, pc2_loadings_heatmap_ax = plt.subplots(figsize=(10, 5))
    im3 = pc2_loadings_heatmap_ax.imshow(pc2_sorted_trace.T, cmap='viridis', aspect='auto')
    pc2_loadings_heatmap_fig.colorbar(im3, ax=pc2_loadings_heatmap_ax, label='Activity')
    pc2_loadings_heatmap_ax.set_xlabel('Time')
    pc2_loadings_heatmap_ax.set_ylabel('Neurons (sorted by PC2 loading)')
    pc2_loadings_heatmap_ax.set_title(f'Heatmap (PC2 sorted) - {condition_name} agarose')
    pc2_loadings_heatmap_fig.tight_layout()

    # ── Behavior Correlation Heatmap ──────────────────────────────────────────
    col_corr_pairs = list(zip(cond_trace_df.columns, cond_correlations))
    col_corr_pairs.sort(key=lambda x: -2.0 if np.isnan(x[1]) else x[1])  # sort ascending correlation, NaNs at bottom
    sorted_cols_corr = [c for c, _ in col_corr_pairs]
    corr_sorted_trace = cond_trace_df[sorted_cols_corr]

    beh_corr_heatmap_fig, beh_corr_heatmap_ax = plt.subplots(figsize=(10, 5))
    im4 = beh_corr_heatmap_ax.imshow(corr_sorted_trace.T, cmap='viridis', aspect='auto')
    beh_corr_heatmap_fig.colorbar(im4, ax=beh_corr_heatmap_ax, label='Activity')
    beh_corr_heatmap_ax.set_xlabel('Time')
    beh_corr_heatmap_ax.set_ylabel('Neurons (sorted by Behavior Correlation)')
    beh_corr_heatmap_ax.set_title(f'Heatmap (Behavior Correlation sorted) - {condition_name} agarose')
    beh_corr_heatmap_fig.tight_layout()

    # ── Ranked contribution table ─────────────────────────────────────────────
    # Express each loading as % of total absolute loadings, then cumulative sum
    abs_total = sum(abs(c) for c in cond_components_pc1)
    contrib_pct = [abs(c) / abs_total * 100 for c in cond_sorted_components]
    cumulative  = []
    running = 0
    for pct in contrib_pct:
        running += pct
        cumulative.append(round(running, 2))

    contrib_table = pd.DataFrame({
        'Neuron':               cond_sorted_names,
        'PC1 loading':          [round(c, 4) for c in cond_sorted_components],
        'Contribution (%)':     [round(p, 2) for p in contrib_pct],
        'Cumulative sum (%)':   cumulative,
    })
    print(f'\n=== PC1 Contributions — {condition_name} agarose ===')
    print(contrib_table.to_string(index=False))

    # Record PC1 loading for every neuron (identified and unidentified)
    for name, loading in zip(cond_neuron_names, cond_components_pc1):
        if name not in pc1_signs_across_conditions:
            pc1_signs_across_conditions[name] = {}
        pc1_signs_across_conditions[name][condition_name] = loading

    # Record dendrogram group assignments (map reordered columns back to group)
    # cond_trace_df columns are already reordered by dendrogram leaves
    reordered_cols = list(cond_trace_df.columns)
    for col_name, neuron_name in zip(reordered_cols, cond_neuron_names):
        # Find original column index to look up dendro_groups
        orig_idx = list(pd.read_hdf(paths['traces']).columns).index(col_name)
        grp = int(dendro_groups[orig_idx])
        all_dendro_group_rows.append({
            'Neuron':    neuron_name,
            'Condition': condition_name,
            'Dendro Group': grp,
        })

    # Save standalone PNG images to save_dir
    figs_to_save = {
        'dendrogram': dend_fig,
        'dendrogram_heatmap': dend_heatmap_fig,
        'pc1_sorted_heatmap': pc1_loadings_heatmap_fig,
        'pc2_sorted_heatmap': pc2_loadings_heatmap_fig,
        'behavior_corr_heatmap': beh_corr_heatmap_fig,
        '3d_pca': fig_3d,
        '2d_pca': fig_2d,
        'scree_normal_pca': scree_norm_fig,
        'scree_transposed_pca': scree_fig,
        'transposed_scatter_pc1_cluster': scatter_t_fig,
        'transposed_scatter_pc2_loading': scatter_t2_fig,
        'kmeans_elbow_pc1': elbow_fig,
        'pc1_vs_neurons_behavior': bx_fig,
        'pc1_contribution_bars': bar_fig,
        'pc2_contribution_bars': pc2_bar_fig
    }
    for fig_name, fig_obj in figs_to_save.items():
        fig_obj.savefig(os.path.join(save_dir, f'{condition_name}_{fig_name}.png'), dpi=300, bbox_inches='tight')

    # Save to PDF and close individual figures to save memory
    pdf_writer.savefig(dend_fig)
    pdf_writer.savefig(dend_heatmap_fig)
    pdf_writer.savefig(pc1_loadings_heatmap_fig)
    pdf_writer.savefig(pc2_loadings_heatmap_fig)
    pdf_writer.savefig(beh_corr_heatmap_fig)
    pdf_writer.savefig(fig_3d)
    pdf_writer.savefig(fig_2d)
    pdf_writer.savefig(scree_norm_fig)
    pdf_writer.savefig(scree_fig)
    pdf_writer.savefig(scatter_t_fig)
    pdf_writer.savefig(scatter_t2_fig)
    pdf_writer.savefig(elbow_fig)
    pdf_writer.savefig(bx_fig)
    pdf_writer.savefig(bar_fig)
    pdf_writer.savefig(pc2_bar_fig)
    plt.close(dend_fig)
    plt.close(dend_heatmap_fig)
    plt.close(pc1_loadings_heatmap_fig)
    plt.close(pc2_loadings_heatmap_fig)
    plt.close(beh_corr_heatmap_fig)
    plt.close(fig_3d)
    plt.close(fig_2d)
    plt.close(scree_norm_fig)
    plt.close(scree_fig)
    plt.close(scatter_t_fig)
    plt.close(scatter_t2_fig)
    plt.close(elbow_fig)
    plt.close(bx_fig)
    plt.close(bar_fig)
    plt.close(pc2_bar_fig)

pdf_writer.close()

print(f"\nAll plots have been completely saved to '{pdf_path}' instead of popup windows!")

# ── Categorize neurons and export to Excel ───────────────────────────────
rows = []
for neuron, loadings in pc1_signs_across_conditions.items():
    is_identified = not neuron.startswith('neuron_')
    v1 = loadings.get('1per')
    v2 = loadings.get('2per')
    v3 = loadings.get('3per')
    values = [v for v in [v1, v2, v3] if v is not None]
    if all(v > 0 for v in values):
        category = 'Always Positive'
    elif all(v < 0 for v in values):
        category = 'Always Negative'
    else:
        category = 'Mixed or Zero'
    rows.append({
        'Neuron':         neuron,
        'Is Identified':  is_identified,
        '1per PC1 loading': v1,
        '2per PC1 loading': v2,
        '3per PC1 loading': v3,
        'Category':       category,
    })

category_order = {'Always Positive': 0, 'Always Negative': 1, 'Mixed or Zero': 2}
result_df = pd.DataFrame(rows)
result_df = result_df.sort_values(
    by=['Category', 'Is Identified', 'Neuron'],
    key=lambda col: col.map(category_order) if col.name == 'Category' else col
).reset_index(drop=True)

excel_path = f'pc1_neuron_categories_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
cluster_df = pd.DataFrame(all_cluster_rows)

# ── Build consistent-cluster sheet ──────────────────────────────────────────
# For each neuron, check which cluster it belongs to across all conditions it appears in
neuron_cluster_consistency = {}
for row in all_cluster_rows:
    n = row['Neuron']
    if n not in neuron_cluster_consistency:
        neuron_cluster_consistency[n] = []
    neuron_cluster_consistency[n].append(row['Cluster'])

consistency_rows = []
for neuron, clusters in neuron_cluster_consistency.items():
    is_identified = not str(neuron).startswith('neuron_')
    unique_clusters = set(clusters)
    if len(unique_clusters) == 1:
        only_cluster = unique_clusters.pop()
        if only_cluster == 0:
            group = 'reverse neuron - always positive cluster'
        elif only_cluster == 2:
            group = 'forward neuron - always negative cluster'
        else:
            group = 'Only Cluster 1 (middle)'
    else:
        group = 'Mixed clusters'
    consistency_rows.append({'Neuron': neuron, 'Is Identified': is_identified, 'Clusters across conditions': str(sorted(clusters)), 'Group': group})

consistency_df = pd.DataFrame(consistency_rows)
group_order = {'reverse neuron - always positive cluster': 0, 'Only Cluster 1 (middle)': 1, 'forward neuron - always negative cluster': 2, 'Mixed clusters': 3}
consistency_df = consistency_df.sort_values(
    by=['Group', 'Neuron'],
    key=lambda col: col.map(group_order) if col.name == 'Group' else col
).reset_index(drop=True)

# Build dendrogram groups Excel sheet
dendro_df = pd.DataFrame(all_dendro_group_rows)
corr_df = pd.DataFrame(all_corr_rows)

with pd.ExcelWriter(excel_path) as writer:
    result_df.to_excel(writer, sheet_name='PC1 Categories', index=False)
    cluster_df.to_excel(writer, sheet_name='KMeans Clusters', index=False)
    consistency_df.to_excel(writer, sheet_name='Cluster Consistency', index=False)
    dendro_df.to_excel(writer, sheet_name='Dendro Groups', index=False)
    corr_df.to_excel(writer, sheet_name='Behavior Correlation', index=False)
    print(f'\nNeuron categories saved to: {excel_path}')
print(result_df.groupby('Category')['Neuron'].count().to_string())
print(f'\nKMeans cluster counts per condition:')
print(cluster_df.groupby(['Condition', 'Cluster'])['Neuron'].count().to_string())
print(f'\nCluster consistency groups:')
print(consistency_df.groupby('Group')['Neuron'].count().to_string())
print(f'\nDendrogram group counts per condition:')
print(dendro_df.groupby(['Condition', 'Dendro Group'])['Neuron'].count().to_string())
