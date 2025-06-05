
# %%
import pyvista as pv 
import pygeodesic.geodesic as geodesic


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# Path to your FreeSurfer surface file (e.g., lh.white)
surf_path_pial = r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\exampleData\HCP_rawSurfaces\100206\surf\lh.pial"
surf_path_white = r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\exampleData\HCP_rawSurfaces\100206\surf\lh.white"
surf_path_inflated = r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\exampleData\HCP_rawSurfaces\100206\surf\lh.inflated"


# Load surface
coords_white, faces_white = nib.freesurfer.read_geometry(surf_path_white)
coords_pial, faces_pial = nib.freesurfer.read_geometry(surf_path_pial)
coords_inflated, faces_inflated = nib.freesurfer.read_geometry(surf_path_inflated)


# timeseries = np.load(r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\output\wave_output_decay.npy")
# timeseries = np.load(r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\output\wave_output_decay3.npy")
# timeseries = np.load(r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\output\wave_output_decay13.npy")

# %%

label_file_path = r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\exampleData\HCP_rawSurfaces\100206\label\lh.aparc.annot"
labels, ctab, names = nib.freesurfer.read_annot(label_file_path)

coords = (coords_pial + coords_white)/2
faces = faces_pial

print(f"Loaded annotation file: {label_file_path}")
print(f"Number of labels: {len(np.unique(labels))}")
print(f"Label names: {names[:10]}")  # Print first 10 label names

# Each entry in 'labels' is the label index for the corresponding vertex in the mesh
# 'ctab' is the color table, 'names' are the label names

# ...existing code...

# For each label, calculate the mean vertex coordinate from coords
unique_labels = np.unique(labels)
label_mean_coords = {}
label_seed_vertex = {}

for label in unique_labels:
    vertex_indices = np.where(labels == label)[0]
    mean_coord = coords[vertex_indices].mean(axis=0)
    label_mean_coords[label] = mean_coord
    # Find the vertex index closest to the mean coordinate
    distances = np.linalg.norm(coords[vertex_indices] - mean_coord, axis=1)
    closest_vertex = vertex_indices[np.argmin(distances)]
    label_seed_vertex[label] = closest_vertex
    print(f"Label {label}: mean vertex coordinate {mean_coord}, seed vertex {closest_vertex}")

# Optionally, if you want to map label indices to label names:
label_names = {label: names[i].decode('utf-8') for i, label in enumerate(unique_labels)}
for label in unique_labels:
    print(f"Label {label} ({label_names[label]}): mean vertex coordinate {label_mean_coords[label]}")

# ...existing code...

# %% Create a synthetic wave on the cortical surface mesh
# Wave parameters
wave_speed = 3.5 # Speed of wave in cortex (mm/min) from Hadjikhani et al. 2001

# Number of time steps (in generations)
time = np.arange(0, 101, 1)

# Set thickness of wave (in generations)
time_delta = 1.5

# Label index to start the wave from
label_index = 13 # lingual

# Calculate geodesic distances using PyGeodesic
geoalg = geodesic.PyGeodesicAlgorithmExact(coords, faces)
# sourceIndex = np.array([label_seed_vertex[unique_labels[label_index]]])  # Seed vertex for the wave
sourceIndex = np.array([label_seed_vertex[13]]) 
# sourceIndex = np.array([0])
targetIndex = None
distance,path = geoalg.geodesicDistances(sourceIndex, targetIndex) # (in mm)

# Change time from generations to minutes
time = time / 5  # Convert generations to minutes (assuming each generation is 0.1 min)

# Make time and distance 2D arrays for broadcasting
time = np.atleast_2d(time)  # Convert time to a column vector
distance = np.atleast_2d(distance).T  # Convert distance to a column vector

# Change distance to time on (equal to distance / wave_speed)
time_on = distance / wave_speed  # Convert distance to time (in mins)

 # Create a time on by time matrix
output = time*time_on


# If time is less than time on, set output to 0
output[((time<time_on))] = 0

# If time is great than time on, and less than time on + time_delta, set output to 1
output[((time>=time_on) & (time<time_on + time_delta))] = 1

# If time is greater than time on + time_delta, set output to exponential decay
time_decay = 2  # Adjust this value to control the decay time (min)
#output[(time>=(time_on + time_delta))] = np.exp(-(time-(time_on + time_delta)) / time_decay)

# Compute the decay for all elements
decay = np.exp(-(time - (time_on + time_delta)) / time_decay)
# Assign only where the mask is True
decay_mask = (time >= (time_on + time_delta))
output[decay_mask] = decay[decay_mask]


# %%

timeseries = output
# timeseries = np.atleast_2d(labels).T

# use pyvista to create a mesh
# Convert faces to the format expected by pyvista: (N, 4) with leading 3s
faces_pv = np.hstack([np.full((faces_white.shape[0], 1), 3), faces_white]).astype(np.int64)
mesh_pial = pv.PolyData(coords_pial, faces_pv)
mesh_white = pv.PolyData(coords_white, faces_pv)
mesh_inflated = pv.PolyData(coords_inflated, faces_pv)
mesh_mid = pv.PolyData((coords_pial + coords_white)/2, faces_pv)

# view the mesh
plotter = pv.Plotter(window_size=[1920, 1200])
plotter.add_mesh(mesh_pial, name='brainmesh', scalars=timeseries[:,0], cmap='viridis', show_scalar_bar=False)

# set default camera position
plotter.camera_position = 'yz'
plotter.camera_position = [(-300, -35, -3), (-32, 0, 0), (0, 0, 1)]  # Adjust the camera position as needed

# add a slider to change the vertex generations
def update_mesh(value,mesh):
    plotter.add_mesh(mesh, name='brainmesh', scalars=timeseries[:, int(value)], cmap='viridis', show_scalar_bar=False)

# mesh = mesh_pial.copy()  # Start with the pial mesh
# mesh = mesh_white.copy()  # Start with the white mesh
# mesh = mesh_inflated.copy()  # Start with the inflated mesh
mesh = mesh_mid

# Add a slider to the plotter
plotter.add_slider_widget(
    lambda value: update_mesh(value, mesh),
    [0, timeseries.shape[1] - 1],
    title='Time',
    value=0,
    pointa=(0.1, 0.9, 0),
    pointb=(0.9, 0.9, 0),
    interaction_event='always'
)


# Show the plotter with the slider
plotter.show()

# %% print the final camera position
print("Final camera position:", plotter.camera_position)