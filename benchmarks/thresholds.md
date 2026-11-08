# Threshold evidence

## Measurements

| Host | Tier | Cell | Arm | Library SHA | Session | Unstable | ns/call |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Caveats

- Host order: Willow Cove, Golden Cove.
- Accepted results publish the candidate arm; a session whose two arms share one library SHA is labelled `baseline`. Calibration base-vs-base rows are labelled `baseline`, and layout-control candidates are labelled `layout_control`.
- Aggregation: each row reports the arm's session summary, computed as the median of its per-run summaries; each run summary is the median of that run's per-round inner-batch medians.
- Setup, allocation, coefficient construction, and input generation stay outside the timed region; inputs are warmed before each timed batch.
- Every cell is checked against the independent element-wise oracle before timing.
- Snapshots pin 64-byte function and loop alignment with `-C llvm-args=-align-all-functions=6` and `-C llvm-args=-align-all-nofallthru-blocks=6`.
- Descriptive records measure one revision without an acceptance claim. Minimum and maximum describe the range across standalone run summaries, not a confidence interval.
- Latest measurements retain their measured revision, session, and harness snapshot. Rows from different sessions are not matched comparisons or proof of the newest release's performance.
- Speedup, regression, crossover, and competitor claims use matched same-session measurements of the relevant cases, not historical row ratios.
- The surfaces harness uses its own warm-calibrate-median sampler and ordinary fixture placement; its timings are not interchangeable with the page-aligned operations harness.
