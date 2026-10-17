# Narrow down remaining multi-contact differences

2026-09-11. CFM=0 did not explain the large discrepancy: native CPU w=1.5/B changed from 132.795 to 132.144 mm and w=2.5/B from 194.838 to 194.655 mm. The large gait gap remains. CFM zero is not adopted as a performance fix.

Three further isolated checks are justified before another expensive GPU grid is selected:

1. Run the complete articulated chain with the same target history, physical PD and numerical step, with both gravity and ground contact disabled. This distinguishes a remaining coupled-drive error from contact behavior; the existing one-DOF and bare inverse-mass tests are insufficient for this particular question.
2. Reduce only the native per-shape contact offset from 20 to 2 µm, keeping friction offset and correlation distance unchanged. These are numerical contact-generation margins, not changes in physical collider radius.
3. Reduce only native friction correlation distance from 50 to 5 µm, preserving the other offsets. This tests contact-anchor persistence/merging separately from contact generation.

Both contact tests retain the two sensitive B-family gaits at their original native environment offsets. They are diagnostic ablations, not new accepted profiles. GPU friction-limit probes will follow the still-running original GPU half-step validation; no overlapping GPU physics job is permitted. All old scores and fixed acceptance gates remain intact.
