# Layout exceptions

## Recorded exceptions

| Session | Host | Tier | Cell | Library SHA | Failed bound (log ratio) | Spread (log ratio) | Disposition | Owner signature |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Caveats

- Host order: Willow Cove, Golden Cove.
- Exception rows record a layout-only failure; they are not passes and their cells are excluded from the current and history tables.
- Failed bounds and spreads are session-rule log ratios; the spread is the largest layout-control upper bound for the cell.
- Inputs are warmed before each timed batch, and setup, allocation, and coefficient construction stay outside the timed region.
- Snapshots pin 64-byte function and loop alignment with `-C llvm-args=-align-all-functions=6` and `-C llvm-args=-align-all-nofallthru-blocks=6`.
