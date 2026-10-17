# Further MuJoCo refinement in the two sensitive gaits

2026-09-11, before the new runs. TGS refinement still shifts 1.5/B and 2.5/B substantially. Although the prescribed MuJoCo 31.25→15.625 µs check was small in the prior clock, a single half-step check does not establish the exact continuous-time limit of a stiff coupled system. Run the same physical MuJoCo v8 implementation at 7.8125 µs, held 250 µs command clock, 6 s, these two unchanged holdouts. This is additional diagnosis, not a new relaxed acceptance rule or a performance-selected timestep.

The separate NoSlip 20-iteration 2.5/B diagnostic completed: 160.341313 mm versus baseline 160.109891 mm at the same 31.25 µs and 250 µs command clock (about +0.145%). It does not explain the large native discrepancy and is not adopted. It generated no MuJoCo warnings.
