
# %%
import pyvista as pv 
import pygeodesic.geodesic as geodesic
import os

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# Path to your FreeSurfer surface file (e.g., lh.white)
surfer_path = r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\exampleData\HCP_rawSurfaces\100206"

hemi = "lh"  # Left hemisphere

# Load surface
coords_white, faces_white = nib.freesurfer.read_geometry(os.path.join(surfer_path,"surf", f"{hemi}.white"))  # Load white surface
coords_pial, faces_pial = nib.freesurfer.read_geometry(os.path.join(surfer_path,"surf", f"{hemi}.pial"))  # Load pial surface
coords_inflated, faces_inflated = nib.freesurfer.read_geometry(os.path.join(surfer_path,"surf", f"{hemi}.inflated"))
coords_inflated = coords_inflated * .55  # Scale down the inflated surface for better visualization
coords_sphere, faces_sphere = nib.freesurfer.read_geometry(os.path.join(surfer_path,"surf", f"{hemi}.sphere"))  # Load sphere surface


# %%

# label_file_path = r"C:\Users\scedg10\OneDrive - Cardiff University\python\MIC-HACK2025_CSD\exampleData\HCP_rawSurfaces\100206\label\lh.aparc.annot"
label_file_path = os.path.join(surfer_path,"label", f"{hemi}.aparc.annot")
labels, ctab, names = nib.freesurfer.read_annot(label_file_path)

coords = (coords_pial + coords_white)/2
faces = faces_pial

print(f"Loaded annotation file: {label_file_path}")
print(f"Number of labels: {len(np.unique(labels))}")
print(f"Label names: {names[:10]}")  # Print first 10 label names

# Each entry in 'labels' is the label index for the corresponding vertex in the mesh
# 'ctab' is the color table, 'names' are the label names

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


# %% Create a synthetic wave on the cortical surface mesh
# Wave parameters
# wave_speed = 3.5 # Speed of wave in cortex (mm/min) from Hadjikhani et al. 2001
wave_speed = 2

# Time range for the wave (in minutes)
timeMax = 30  # Maximum time in minutes
time_mins = np.linspace(0, timeMax, 100)

# Set thickness of wave (in minutes)
time_delta = 0.75

# Label index to start the wave from
# label_index = 14 # lingual
label_index = 22 # pericalcarine

# Calculate geodesic distances using PyGeodesic
geoalg = geodesic.PyGeodesicAlgorithmExact(coords, faces)
# sourceIndex = np.array([label_seed_vertex[unique_labels[label_index]]])  # Seed vertex for the wave
sourceIndex = np.array([label_seed_vertex[np.where(unique_labels == label_index)[0][0]]])  # Seed vertex for the wave
# sourceIndex = np.array([0])
targetIndex = None
distance_mm,path = geoalg.geodesicDistances(sourceIndex, targetIndex) # (in mm)

# Make time and distance 2D arrays for broadcasting
time_mins = np.atleast_2d(time_mins)  # Convert time to a column vector
distance_mm = np.atleast_2d(distance_mm).T  # Convert distance to a column vector

# Change distance to time on (equal to distance / wave_speed)
time_on = distance_mm / wave_speed  # Convert distance to time (in mins)

 # Create a time on by time matrix
timeseries = time_mins * time_on

# If time is less than time on, set output to 0
timeseries[((time_mins<time_on))] = 0

# If time is great than time on, and less than time on + time_delta, set output to 1
timeseries[((time_mins>=time_on) & (time_mins<time_on + time_delta))] = 1

# If time is greater than time on + time_delta, set output to exponential decay
time_decay = 2  # Adjust this value to control the decay time (min)

# Compute the decay for all elements
decay = np.exp(-(time_mins - (time_on + time_delta)) / time_decay)
# Assign only where the mask is True
decay_mask = (time_mins >= (time_on + time_delta))
timeseries[decay_mask] = decay[decay_mask]


# %%


# use pyvista to create a mesh
# Convert faces to the format expected by pyvista: (N, 4) with leading 3s
faces_pv = np.hstack([np.full((faces_white.shape[0], 1), 3), faces_white]).astype(np.int64)

mesh = pv.PolyData((coords_pial + coords_white)/2, faces_pv)

# view the mesh
plotter = pv.Plotter(window_size=[1920, 1000])

# set default camera position
plotter.camera_position = 'yz'
# plotter.camera_position = [(-300, -35, -3), (-32, 0, 0), (0, 0, 1)]  # Adjust the camera position as needed
plotter.camera_position = [(220, 0, 0), (-32, 0, 0), (0, 0, 1)]  # Adjust the camera position as needed


mesh.point_data['timeseries'] = timeseries[:, 0]  # Initialize with the first time point


# add a slider to change the vertex generations
def update_mesh(value):
    global mesh
    mesh.point_data['timeseries'] = timeseries[:, int(value)]
    update_view()

def update_mesh_coords(value):
    # Update the mesh coordinates based on the value
    if value < 0 or value > 3:
        raise ValueError("Value must be between 0 and 3")
    if value < 1:
        new_coords = coords_white + (coords_pial - coords_white) * value
    elif value < 2:
        new_coords = coords_pial + ((coords_inflated) - coords_pial) * (value - 1)
    else:
        new_coords = coords_inflated + (coords_sphere - coords_inflated) * (value - 2)
    global mesh
    mesh.points = new_coords
    update_view()
    
def update_view():
    # plotter.add_mesh(mesh, name='brainmesh', scalars='timeseries', cmap='viridis', show_scalar_bar=False,smooth_shading=True, show_edges=True)
    plotter.add_mesh(mesh, name='brainmesh', scalars='timeseries', cmap='viridis', show_scalar_bar=False,smooth_shading=False, show_edges=False)


update_view()

# Add a slider to the plotter
plotter.add_slider_widget(
    lambda value: update_mesh(value),
    [0, timeseries.shape[1] - 1],
    title='Time',
    value=0,
    pointa=(0.1, 0.9, 0),
    pointb=(0.9, 0.9, 0),
    interaction_event='always',
    title_height=.03,
    style='modern'
)

# Add a slider to the plotter
plotter.add_slider_widget(
    lambda value: update_mesh_coords(value),
    [0, 2],
    # title='Surface Morphing',
    title = '',
    value=0,
    pointa=(0.1, 0.05, 0),
    pointb=(0.1, 0.9, 0),
    interaction_event='always',
    title_height= .03,
    style='modern')

# plotter.enable_depth_of_field()
# pl.camera.zoom(1.5)
# plotter.enable_anti_aliasing('ssaa')


# Show the plotter with the slider
plotter.show()

# %% print the final camera position
print("Final camera position:", plotter.camera_position)