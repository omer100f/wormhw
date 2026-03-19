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
from scipy.spatial.distance import squareform

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

path_to_traces = r".\1per\2024-07-12_15-57_1per_worm2-2024-07-12_ratio.h5"
path_to_behavioral_annotations_1per  = r".\1per\2024-07-12_15-57_1per_worm2-2024-07-12_beh_annotation.csv"
beh_annotations_df = pd.read_csv(path_to_behavioral_annotations_1per, header=None)
beh_annotations_df = beh_annotations_df.iloc[::24]
# downsample beh_annotations_df index by 24
beh_annotations_df.index = beh_annotations_df.index // 24


trace_df = pd.read_hdf(path_to_traces)

# dataframe format is:
# every row is a time point, every column is a neuron

# plot an interesting trace
trace_df['neuron_025'].plot()

# this is a trace of a reversal neuron that is active when worms reverse
# How do i know this? Becuase we IDd all the neurons we know how to ID in this dataset
# to open these ids
neuron_id_path = r".\1per\2024-07-12_15-57_1per_worm2-2024-07-12_neuron_ids.xlsx"
neuron_id_df = pd.read_excel(neuron_id_path)

#the ID of the neurons is kept in 'ID1' column
#which neurons do we have?
print(neuron_id_df['ID1'].unique())
"""
'RMEL' 'VB02' nan 'OLQDR' 'IL1L' 'SIAVR' 'RMER' 'URADL' 'URADR' 'URYVR'
 'URYVL' 'BAGL' 'IL2DL' 'IL2L' 'AVAL' 'RMDVL' 'SAAVL' 'FLPL' 'URYDL'
 'OLQDL' 'SIAVL' 'RMED' 'M3L' 'URYDR' 'M4' 'SMDVR' 'IL2VL' 'DB02' 'VA01'
 'AIBL' 'RIAL' 'SIADL' 'AINL' 'M3R' 'AINR' 'AVAR' 'AVBR' 'RMDDR' 'RIBL'
 'ASGR' 'URXL' 'NSMR' 'NSML' 'MI' 'RIVL' 'I1R' 'RIMR' 'SMDVL' 'M5' 'RMEV'
 'RIVR' 'I2L' 'AQR' 'RIS' 'I3' 'AVER' 'OLQVR' 'CEPVR' 'FLPR' 'RMDVR'
 'DB01' 'SMDDL' 'RID' 'OLQVL' 'I2R' 'I1L' 'RMGL' 'M1' 'I4' 'I6' 'URXR'
 'ALA' 'IL2DR' 'IL2VR' 'BAGR' 'ASGL' 'RIBR' 'SIADR' 'SMDDR' 'RIAR' 'SAAVR'
 'AVBL' 'ADAL' 'IL1R' 'IL2R' 'DD01' 'AVE`L' 'RMGR' 'AIBR' 'ADAR' 'ADER'
 'RMDDL' 'RIML_' 'DA01' 'ADEL' 'VB03' 'VA02'
 """

### Step 2 - Further instructions ###
# To further investigate the data lets visualize the whole brain using a heatmap, together with the behavior annotations
# lastly will use principal component analysis to see if we can find some interesting patterns in the data, when compared to the behavior annotations.

# visualize the data using a heatmap, together with the behavior annotations
# first we need to interpolate the data
# then we need to order the neurons based on how similar they are to each other, so we can see if there are any clusters of neurons that have similar activity patterns
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
trace_df = pd.read_hdf(path_to_traces)
trace_df = trace_df.interpolate(method='linear', axis=0, limit_direction='both')

# first we need to order the neurons based on how similar they are to each other
# used squareform on to make the matrix into a 1D array otherwise the linkage thinks its a raw data matrix instead of distances
distance_array = squareform(pairwise_distances(trace_df.T, metric='correlation')) # use correlation distance to measure similarity between neurons
linkage_matrix = linkage(distance_array, method='average', metric='correlation') # use average linkage to cluster the neurons based on their similarity
dendrogram = dendrogram(linkage_matrix, no_labels=True, no_plot=True) # get the order of the neurons based on the dendrogram
trace_df = trace_df.iloc[:,dendrogram['leaves']] # reorder the neurons based on the order of the dendrogram

# plot the heatmap
plt.imshow(trace_df.T, cmap='viridis', aspect='auto',filterrad=0.00001)
# add color map
plt.colorbar(label='Activity')
plt.xlabel('Time')
plt.ylabel('Neurons')

# make a principal component analysis of the data

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(trace_df)

# Get lists of time points according to the worm behavior
reverse_time_points = list(map(int, beh_annotations_df[beh_annotations_df[1] == 1].index.tolist()))
forward_time_points = list(map(int, beh_annotations_df[beh_annotations_df[1] == -1].index.tolist()))
stay_time_points    = list(map(int, beh_annotations_df[beh_annotations_df[1] == 0].index.tolist()))

# plot the first 3 principal components but as a line plot in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pca_result[forward_time_points,0], pca_result[forward_time_points,1], pca_result[forward_time_points,2], color='blue')
ax.plot(pca_result[reverse_time_points,0], pca_result[reverse_time_points,1], pca_result[reverse_time_points,2], color='red')
ax.plot(pca_result[stay_time_points,0], pca_result[stay_time_points,1], pca_result[stay_time_points,2], color='green')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Task 1: TODO: color the 3d line plot based on the behavior annotations,
#  so we can see if there are any patterns in the data that are related to the behavior of the worm.
#  For example, we can color the line red when the worm is reversing and blue when it is going forward. T
#  his way we can see if there are any clusters of activity in the principal component space that are related to the behavior of the worm.

# Task 2: TODO: plot the first principle component versus the activity of AVA, which is a reversal neuron, to see if there is any correlation between the two.
bx = plt.figure().add_subplot()
bx.plot(pca_result[:, 0], label='PC1', color='blue')
bx.plot(trace_df[neuron_id_df.loc[neuron_id_df['ID1'] == "AVAL", 'Neuron ID']], color='black', label='AVAL')
bx.plot(trace_df[neuron_id_df.loc[neuron_id_df['ID1'] == "AVAR", 'Neuron ID']], color='gray', label='AVAR')
bx.legend(loc='upper right')

# Task 3: TODO: look up the contribution of every identified neuron to the first principal component,
# and see if there are any interesting neurons that contribute a lot to the first principal component,
# which is the one that explains the most variance in the data. This can be done by looking at the components_ attribute of the PCA object,
# which gives the contribution of each original feature (neuron) to each principal component.
# We can then plot these contributions as a bar plot to see which neurons contribute the most to the first principal component.
# Note: You will need to add the neuron IDs to the dataframe to do that!
components_pc1 = pca.components_[0].tolist()

# Get neuron names from neuron_id_df. using the original column name if no ID1
id_mapping = dict(zip(neuron_id_df['Neuron ID'], neuron_id_df['ID1']))
neuron_names = [id_mapping.get(col, col) if pd.notna(id_mapping.get(col, col)) else col for col in trace_df.columns]

# Sort the neurons based on their contribution to PC1
sorted_pairs = sorted(zip(neuron_names, components_pc1), key=lambda x: x[1])
sorted_neuron_names, sorted_components_pc1 = zip(*sorted_pairs)

pc1_components_graph = plt.figure().add_subplot()
pc1_components_graph.set_xlabel('neurons')
pc1_components_graph.set_ylabel('contribution to pc1')
pc1_components_graph.bar(sorted_neuron_names, sorted_components_pc1, color='blue')
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.show()
# Task 4: TODO: compare the activity across the agarose conditions, and the contributions of the neurons to the first principal component.

