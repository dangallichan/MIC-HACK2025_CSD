# MIC-HACK2025_CSD
Can we find evidence of CSD in resting-state fMRI datasets?

We got as far as getting a reasonable simulation to work that spreads from a brain region of choice along a Freesurfer surface. Currently the spreading is calculated by distance along a mesh at the average point between the white surface and the pial surface. PyVista is used for visualisation, which we found seemed to work well on Windows, Mac and Linux - and sliders can be added without too much fuss. Additional functionality that might be useful (buttons, drop-downs) would still be intricate - but possible extensions!

Packages used:
- Python 3.10
- PyVista
- PyGeodesic (to calculate distances along mesh)

This is `test_pyvista_directSim2.py` - and it just requires a Freesurfer output folder to work:

https://github.com/user-attachments/assets/5758c097-a1d3-46e7-bdd4-9f8a670ae26e


***

We also experimented with using 'raw' VTK (see `vis1_usingVTK.py`), but this can end up requiring a lot more code to create the desired features.
