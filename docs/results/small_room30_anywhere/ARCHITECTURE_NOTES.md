# History- and Purpose-Conditioned Affordance Weighting

## Figure caption

Overview of the Teacher–Student affordance pipeline. A frozen ADM+LoRA Teacher supplies a six-channel base affordance map. The Student predicts scene-only candidate slots, context anchors, and point-to-slot support assignments. Separate relation and history mixture-of-experts branches predict candidate-level scores from purpose text and observed history, respectively. Their scores are combined as S_i = 0.7 R_i + 0.3 H_i, and candidate preferences are obtained by softmax with temperature 0.1. These preferences are projected through scene-only support assignments into a point-wise weight shared across body channels. Multiplication with the Teacher map produces the conditioned affordance map. Training annotations supervise Student components but are not inference inputs. The downstream motion module denotes proposed integration, not an evaluated result of this experiment.

## Scientific scope

- N = 8192 scene points; six body channels; K = 2 candidate slots plus a null support channel.
- Observed history: eight frames spanning 0–0.35 seconds, with XY position, velocity, and heading.
- Teacher action text for these sitting maps: “Sit on something”.
- Student purpose conditions include unrestricted sitting, watching TV, writing near a desk or whiteboard, desk-specific writing, and whiteboard-specific writing. Applicability varies by room.
- R and H are candidate-level scores, not point-wise spatial maps.
- q is a relative candidate preference, not calibrated correctness probability.
- w_n = sum_i B_ni q_i, excluding the null channel. The same w_n multiplies every body channel.
- The map weights contact/support regions; it does not predict a path from the observed starting position.
- The anywhere target assigns R* = 1 to both valid sitting candidates, allowing history to distinguish them.
- Current same-room evaluation does not establish unseen-room generalization or end-to-end motion quality.

## Generation brief

Method: built-in image generation, editing the supplied architecture image.

Redesign the reference as a high-resolution English-language academic flowchart on a white background, with thin arrows and restrained blue, lavender, peach, green, and rose panels. Preserve the overall Teacher / two-branch Student / supervision / motion organization while replacing the obsolete 0.6/0.4 point-wise fusion with candidate scores, 0.7/0.3 fusion, temperature-0.1 softmax, and scene-only spatial projection. Keep scene-only candidate perception and support independent of purpose/history. Show the frozen Teacher bypass to final multiplication. Keep training labels outside inference inputs. Mark downstream AMDM motion integration as proposed and not evaluated here. Avoid crossing arrows, decorative effects, and tiny text. Use the caption and scientific scope above as authoritative content.
