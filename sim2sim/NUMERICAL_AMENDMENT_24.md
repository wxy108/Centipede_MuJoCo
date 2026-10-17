# Translation-invariance diagnostic for the remaining lateral residual

The shared-contact main bank passes forward-displacement comparisons by a wide margin, but full motion analysis finds up to 0.751 mm CPU / 0.869 mm GPU relative-centroid XY path difference from Mu over 2–6 s, mainly lateral. These values are retained rather than declaring every trajectory identical. See audit/common_contact_motion_comparison.json.

Before this diagnostic, specify two copies of the same 4.5/A gait in a single CPU process, at global Y = 0 and 1.75 m. Both use the frozen COMMON_CONTACT_V2 force law, canonical asset/PD, PGS16/4, dt31.25 µs, command250 µs, identical root Z, six seconds, and disabled native collision response. Flat-plane physics is translationally invariant in Y. The 1.75 m copy also controls against the already completed main-bank environment 7; if it does not reproduce that trace, batch/actor-layout effects remain a confound and the origin difference must not be assigned a unique cause.

This is a diagnostic, not a replacement validation profile. No origin, gate, gait, PD, material constant or time step of the running main/refinement banks is changed. In particular, do not use this extra run to replace any unfavourable original holdout.
