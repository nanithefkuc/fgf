# Competitor measurements

## Caveats

- Host order: Willow Cove, Golden Cove.
- Competitor values are the harnesses' reported per-arm medians; every arm validates its output before timing.
- Each arm warms immediately before its timed batch; layout conversion, allocation, and input generation stay outside the timed region.
- Throughput columns count one operand pair per element; scalar columns report ns/op.
- Snapshots pin 64-byte function and loop alignment with `-C llvm-args=-align-all-functions=6` and `-C llvm-args=-align-all-nofallthru-blocks=6`.
