# Teacher-v10.3.1 PASS affordance-map viewer

Read-only Viser viewer for the validated Teacher-v10.3.1 two-scene K=3
calibration result. It reads the saved `summary.json` and
`onpolicy_calibration_maps.npz`; it does not run the model or diffusion.

The six panels display Input RGB, all-sittable/per-object GT, the sealed v10.3
start map, a selected calibration update, update-minus-start, and absolute
candidate error. Scene, update, generation, prompt, object, joint channel and
render mode are interactive. Update 1 is the default because the validated
shortlist order is `[1, 2]`.

No optimizer, checkpoint, room_0201 array, or paper-test payload is loaded.
Continuous XY heatmaps are display-only interpolation over the original 8192
point values.
