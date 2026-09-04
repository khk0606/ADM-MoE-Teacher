# Teacher-v9.2 hotspot/trust gate

Fresh v5r4 + zero-init LoRA response grid for correcting the v9.1 all-sittable
failure.  See `TEACHER_V92_HOTSPOT_TRUST_RUNBOOK.md` for Ubuntu commands.

The package changes the optimization signal, not the accepted v9 GT:

- weak support loss prevents uniform object inflation;
- conservative hotspot margin and listwise ordering learn the motion-contact
  shape within Bed, normal Chair and High Chair separately;
- Base trust outside verified objects prevents scene-wide drift;
- absolute explicit-negative loss suppresses TV, Desk and Whiteboard;
- prompt and v5 preservation remain active;
- unknown chairs remain ignored.

No checkpoint is saved by this gate.
