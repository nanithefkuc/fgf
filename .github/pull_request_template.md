Refs #68

## Checklist for SIMD-kernel, routing, and ledger PRs

- [ ] Obligation IDs identified with before/after route and performance state.
- [ ] Independent correctness references (differential, public-route,
  production-smoke case IDs).
- [ ] Direct and public-route native execution evidence under default and all
  features.
- [ ] Pinned interleaved before/after measurements with unchanged controls.
- [ ] Dispatch gates exercised (tier requests and feature configurations).
- [ ] Docs, CHANGELOG, residue-ledger, and applicable unsafe-code updates.