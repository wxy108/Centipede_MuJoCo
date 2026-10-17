# Compliant-tip PD and phase-collocation optimization

This experiment continues the latest compliant-tip Centipede Model. The rigid
tip remains an ablation only and is not used here. The Centipede Robot project
is read-only controller evidence; Robot Step 18/19 data are not read as model
results and are never written.

Fixed: model XML, compliant-tip stiffness/damping, masses, inertias, contact,
flat ground, near-ground initialization, common three amplitudes, controller
frequency and one coupled body/leg wave number. No feedback, clipping, target
smoothing or special-case stabilizer is introduced.

The Robot controller places body-yaw phases at inter-segment joints and leg
phases at segment centers. For the Model,
`theta_leg[i] = theta_body[i] + f*(2*pi*wave/18)`. Geometry gives `f=0.5`.
The sensitivity/search band is `[0.35, 0.65]`; both leg yaw and leg pitch use
the same segment-center phase so their within-leg phase relation is unchanged.

Search variables are the ten existing body-yaw/body-pitch/body-roll/leg-yaw/
leg-pitch stiffness and damping multipliers plus `f`. Amplitudes are fixed.
The registered local Sobol band is 0.70--1.35 times the prior selected
stiffness and 0.70--1.80 times the prior selected damping, separately for
wave 2.5 and 3.0.

Every original absolute stability/contact/tracking gate remains binding;
leg-yaw tracking RMS <= 10 deg is also explicit. Relative to the matching
previously validated compliant-tip profile, displacement must be >=90%, work
per distance <=125%, and slip, force imbalance, normal impulse and chassis
impulse <=110%. Only all-gates-pass candidates are ranked, by forward
displacement. Final selection requires nominal 12 s and half-timestep 6 s
replays. “Best” means best tested in this finite registered search.

Post-validation amendment: no `f>=0.35` nominal winner passed every long and
half-timestep gate. A single bounded engineering confirmation at `f=0.30`
(60% of the geometric midpoint offset, still nonzero and position-aware) is
therefore allowed, using already tested PD profiles rather than reopening a
new search. It must pass the same unchanged gates and is reported separately
from the original registered band.
