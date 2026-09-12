# OCT Illustration Sample Selection

Selected sample: `raw_2683.npy` from the local Harvard-GF cache.

Purpose: a readable illustration of the voxel volume and orthogonal sections,
not an assessment of diagnostic performance or dataset-wide image quality.

Compared 14 cached volumes using central X/Y sections and an unfolded strip
around the visible shared edge. All previews used original uint8 intensities
with the same grayscale limits of 0-255. See `survey.json` and `survey_*.png`.

Rendered opaque 3D blocks for samples 2615, 3294, 2683 and 3100 with the same
camera, unit voxel spacing and grayscale limits. Selected 2683 qualitatively
for visible layer continuity at the corner and fewer broad dark bands on the
top face than the other rendered candidates. Residual speckle and thin dark
bands are retained. No smoothing, cropping, intensity editing or resampling
was performed. No diagnostic label is asserted for sample 2683.

Final files: `figures/oct_sample_2683/solid/`.
Selected slices: X=100, Y=100, Z=75 (zero-based indices). The en-face depth
was chosen for illustration and is not an anatomical layer segmentation.
The eye-location panel is schematic, not patient-specific registration.
Drive delivery remains deferred at the user's request.

Reproduce:

```powershell
python scripts/render_oct_solid.py --volume "$env:TEMP/gf_vol_cache/raw_2683.npy" --output "figures/oct_sample_2683/solid" --slice-z 75
python scripts/check_oct_edge.py --volume "$env:TEMP/gf_vol_cache/raw_2683.npy" --output "figures/oct_sample_2683/solid"
```
