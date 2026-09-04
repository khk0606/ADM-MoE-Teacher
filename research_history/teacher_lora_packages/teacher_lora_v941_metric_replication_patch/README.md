# Teacher-v9.4.1 metric-first replication

The sealed v9.4 radius `0.01` improved the real top-k membership metric for
Bed, normal Chair, and High Chair, with five of six prompt/object cases
improving and the sixth unchanged. The only failed v9.4 condition was an
auxiliary score-margin surrogate for normal Chair.

This package does not retroactively change that sealed failure. It fixes the
radius before reading three new train-only noise/timestep panels and requires:

- every object to improve in pooled top-k;
- no top-k regression in any of the 18 paired cases;
- at least two improving cases per object;
- prompt, negative, background, active-support, recall and v5 retention.

No checkpoint is saved. room_0102, room_0201, and paper-test arrays remain
unread. A pass authorizes only a separate six-update common-descent response.

