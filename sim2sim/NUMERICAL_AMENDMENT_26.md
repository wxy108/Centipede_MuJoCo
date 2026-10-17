# Half-step check of the isolated origin diagnostic

After the frozen CPU half-step bank completed, its largest relative XY path discrepancy from same-step Mu was 3.550056 mm for 4.5/A at global Y=1.75 m; forward displacement remained within the original numerical gates. The complete amendment-24 dt31 diagnostic reproduced the high-Y bank control bitwise. Moving only Y to zero reduced its maximum XY path difference from 0.750733 mm to 0.003139 mm.

Run the identical two-copy origin diagnostic at dt15.625 microseconds, six seconds, CPU two workers, PGS16/4 and fixed 250 microsecond commands. No physical parameter or target changes. Compare its high-Y copy against the completed CPU fine bank and its low-Y copy against Mu at the same step. This is a deliberately selected worst-case causal diagnostic, not an additional unbiased holdout nor a replacement for the original full bank. Do not change the running GPU validation, primary acceptance gates, or the frozen production runner.
