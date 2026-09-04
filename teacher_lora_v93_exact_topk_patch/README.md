# Teacher-v9.3 exact top-k response gate

This package corrects the v9.2 loss/evaluation mismatch.  Training now defines
each object's target hotspot exactly as the evaluator does: filter GT points at
`>=0.30`, choose `ceil(active_count*0.25)` highest values, then directly swap
false predicted top-k points with missing GT top-k points.

The trusted v9.2 preservation terms remain, while the three candidates vary
actual optimizer step size (`1e-5`, `2e-5`, `4e-5`).  The gate loads only
`room_0101`, performs six updates per fresh candidate and saves no checkpoint.

