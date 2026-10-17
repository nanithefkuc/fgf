#!/usr/bin/env python3
"""Publisher regression: the real `field_maps` producer inventory validates.

Feeds actual `field_maps` bench output through the publisher's maps grammar
(`build_maps` per-record checks plus `_validate_maps_completeness`) so a
grammar change that wrongly requires lane-region or pow-control rows for
scalar-only (`ref2-ref8`, `gf1-gf64`) or bits-only (`gf1-aes8`, `gf1-gf16`,
`gf1-gf32`) pairs fails here instead of in a bench campaign.

Usage: build the bench, capture one full run (or `--list-labels` inventory),
then run this script with the log path:

    cargo build --release --bench field_maps
    ./target/release/deps/field_maps-HASH --list-labels > /tmp/maps-inventory.txt
    python3 benches/test_maps_publisher.py /tmp/maps-inventory.txt
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_publish() -> object:
    path = Path(__file__).with_name("publish.py")
    spec = importlib.util.spec_from_file_location("maps_publish_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["maps_publish_under_test"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} FIELD_MAPS_LOG", file=sys.stderr)
        return 2
    publish = load_publish()
    text = Path(argv[1]).read_text(encoding="utf-8", errors="replace")
    # Campaign logs wrap each family run with its unit marker
    # (`bench-refresh-round` prints `UNIT\t<unit>`); raw bench output
    # starts at the benchmark banner, so prepend the maps unit marker.
    if not text.startswith("UNIT\t"):
        text = "UNIT\tself\n" + text
    parser = publish.LogParser("maps", Path(argv[1]))
    parser.parse(text)

    keys = set()
    for rec in parser.records:
        key = publish.case_key(rec)
        keys.add((key, rec.public))
    campaign = publish.Campaign.__new__(publish.Campaign)
    campaign.available_hosts = ["producer"]
    grouped = {("producer", "v3", "maps"): {rnd: keys for rnd in publish.ROUNDS}}
    campaign._validate_maps_completeness(grouped)

    # Pin the contract the completeness fix establishes: scalar-only and
    # bits-only pairs carry no pow control, and `gf1-gf64` carries no lane
    # region rows (its region rows are the packed-bit class).
    labels = {rec.label for rec in parser.records}
    for pair in ("ref2-ref8", "gf1-gf64", "gf1-aes8", "gf1-gf16", "gf1-gf32"):
        assert not any(
            label.startswith(f"control pow {pair}/") for label in labels
        ), f"unexpected pow control for {pair}"
    assert not any(
        label.startswith("gf1-gf64 ") and "/row-" in label and "-bits" not in label
        for label in labels
    ), "unexpected lane region row for gf1-gf64"
    print(
        f"maps publisher regression: {len(parser.records)} records, "
        f"{len(keys)} cases validate"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
