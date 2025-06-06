# MIC-HACK2025_CSD
Can we find evidence of CSD in resting-state fMRI datasets?

We got as far as getting a reasonable simulation to work that spreads from a brain region of choice along a Freesurfer surface. Currently the spreading is calculated by distance along a mesh at the average point between the white surface and the pial surface. PyVista is used for visualisation, which we found seemed to work well on Windows, Mac and Linux - and sliders can be added without too much fuss. Additional functionality that might be useful (buttons, drop-downs) would still be intricate - but possible extensions!

Packages used:
- Python 3.10
- PyVista
- PyGeodesic (to calculate distances along mesh)


https://github.com/user-attachments/assets/86439c8d-b87e-4938-bf0f-0b837278f16e

