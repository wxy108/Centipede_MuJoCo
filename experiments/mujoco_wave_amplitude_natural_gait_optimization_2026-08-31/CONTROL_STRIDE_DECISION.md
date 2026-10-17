# Control-stride benchmark decision

Status: stride 4 was tested and rejected for the formal search.  Every
`cert_*` pilot row uses stride 1: CPG target updates and MuJoCo integration both
occur at the frozen model timestep of 0.5 ms.

The test kept the physics timestep at 0.5 ms and changed only the interval at
which the 1 Hz open-loop target was refreshed (0.5 ms versus 2 ms).  It used
three deliberately different anchors over the same 6 s / 1.5 s evaluation
window.

| case | stride-1 D (mm) | stride-4 D change | base gate parity | natural gate parity | speedup |
|---|---:|---:|---|---|---:|
| accepted clean, wave 1.5 | 132.970 | +0.330% | yes | yes | 1.09x |
| Isaac amplitude anchor, wave 1.5 | 139.055 | +0.285% | yes | yes | 1.07x |
| rejected high-amplitude control, wave 3 | 156.908 | +0.767% | yes | yes | 1.05x |

The approximation preserved classification in these anchors, but its 5--9%
wall-clock improvement is too small to justify mixing fidelities in the broad
search.  The complete machine-readable comparison is
`outputs/data/control_stride_equivalence.json`.
