# Job-shop scheduling

[简体中文](README.zh-CN.md)

GPU-simulated conditional p-bit updates search feasible priority sequences for static job-shop scheduling. A deterministic classical decoder supplies schedules and original integer makespans. Matched CUDA controls and single-worker OR-Tools CP-SAT show where the thermal update rule helps, ties or loses.

The accepted confirmation covers six inputs, three new seeds and four methods: 72 calls plus three fixed-work replays. The heat bath reaches reference optima in 12 of its 18 calls, with no initial reference hits. Held-out ft10 shows improved hit distributions over the matched controls; both Taillard inputs remain above their references, and single-worker CP-SAT has lower mean complete time on every input.

- [Method, measurements and limitations](report.md)
- [Source and reproduction guide](source-guide.md)
- [All calls and portable witnesses](../../data/jssp/rtx5090/calls.json)

The explicit binary representation contains $J^2M$ assignment entries for $J$ jobs and $M$ machines. The 50 × 15 benchmark therefore has 37,500 entries per replica; this is an encoding-scale endpoint, not a frontier scheduling or physical-device claim. Initialization and deterministic decoding are classical support, and improvements must be assessed against the matched controls.
