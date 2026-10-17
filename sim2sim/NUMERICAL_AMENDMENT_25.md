# Isolated constant-velocity translation check

The high-Y copy of amendment 24 is bitwise identical to the original main-bank control through 2.5 s. The Y=0 copy is already closer to Mu in lateral displacement. This identifies origin dependence, but does not by itself separate position integration from contact-force arithmetic.

Before the following tests, specify a zero-gravity, zero-contact, no-gait interval of 0.5 s for two identical native CPU articulations at Y=0 and 1.75 m. Give both the same constant root linear velocity 0.003 m/s in Y, zero angular velocity and zero joint velocities. All joint targets remain zero because gait onset is at 1 s. Record root position and velocity at 100 Hz and verify the contact force is zero. Expected translation is 1.5 mm. Run matched dt31.25 and dt15.625 µs, with all other properties identical.

Compare measured motion with the analytical constant-velocity trajectory and an independently evaluated float32 repeated-position-addition sequence. This diagnostic changes neither the running validation profile nor its acceptance gates. It is not a substitute for a complete gait or a claim about all causes of the old native-contact gap.

The first dt31 initialization (`_000`) failed before physics: integer zero gravity reached the installed USD Python vector binding as NumPy integers. Retry `_001` uses floating-point zeros with the same intended zero gravity. The failed log remains preserved; no original framework file was edited.
