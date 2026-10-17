# Isolated prescribed-slide normal response

Written before the probe. Use the existing rotation-locked sphere/three-slider plant at the same physical scale, with the TGS external-force correction. Prescribe world X speed 30 mm/s and zero Y by overwriting XY position/velocity before each step in both engines; leave Z free. This is an imposed boundary test, not a freely sliding body and not a new robot controller. Its goal is to examine normal contact response during continuous lateral motion, separate from gait, body inertia, torque or amplitude fitting. Compare MuJoCo at the same step and four times finer. Save full normal motion, not only terminal horizontal displacement.

No corresponding change is adopted into the robot. All physical robot inputs remain frozen.
