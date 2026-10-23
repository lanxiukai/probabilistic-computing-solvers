# Graph and list coloring

[简体中文](README.zh-CN.md)

GPU-simulated p-bit conditional blocks preserve one-hot and allowed-list constraints while minimizing conflicts at a fixed palette. Optional clique symmetry reduction and temperature exchange support the search.

The final panel has 56 original calls and four replays. Heat bath reaches zero in 14 of 18 calls; both 25-color Leighton inputs retain conflicts. Some routes improve on matched plateau, while CPU Tabu is faster on five-color graphs, so no universal thermal-update advantage is established. The largest successful storage model has 46,656 bits, of which 28,026 attach to original conflict factors; replica counts are separate.

- [Model, results, resources and complete bibliography](experiment-report.md)
- [Source and reproduction guide](source-guide.md)
- [Confirmation ledger](../../data/graph-coloring/rtx5090/calls.json)

Clique reduction and temperature exchange are classical support; Tabu is an independent reference. No physical p-bit device measurement is reported.
