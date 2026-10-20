#!/usr/bin/env python3
"""Enumerate fgf-owned stable callables from rustdoc JSON, without source matching.

The input comes from nightly rustdoc with all features. Dependency-owned
re-exports, ``internals``, hidden items, constants, and standard trait
implementations are outside the declared-callable set. Inherent methods on
all element families share their public ``Elem::method`` identity; workload
records additionally identify the concrete field and presentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def canonical_api(path: str) -> str:
    """Normalize root re-exports and definition paths to one callable name."""
    parts = path.removesuffix(" (inherent const)").removeprefix("fgf::").split("::")
    if len(parts) >= 2 and parts[-2][:1].isupper():
        return "::".join(parts[-2:])
    if len(parts) == 1 and parts[0] in {
        "backend", "backend_for", "has_vector_elementwise", "vector_elementwise_min_bytes"
    }:
        return "kernel::" + parts[0]
    return "::".join(parts[-2:])


def declared_callables(document: dict) -> set[str]:
    """Return the public function and owned-trait/inherent-method inventory."""
    if document.get("format_version") != 61:
        raise ValueError(
            "public API inventory requires rustdoc JSON schema 61 "
            "from nightly-2026-09-30"
        )
    index = document["index"]
    root = index[str(document["root"])]
    crate_id = root["crate_id"]
    visited: set[str] = set()
    result: set[str] = set()

    def item(identifier):
        return index.get(str(identifier))

    def visible(value: dict) -> bool:
        return value.get("crate_id") == crate_id and not any(
            "doc" in (encoded := json.dumps(attribute)) and "hidden" in encoded
            for attribute in value.get("attrs", [])
        )

    def methods(identifier, owner: str, trait: bool = False):
        value = item(identifier)
        if not value or not visible(value):
            return
        if "function" not in value["inner"]:
            return
        if trait or value.get("visibility") == "public":
            result.add(canonical_api(f"{owner}::{value['name']}"))

    def visit(identifier, path: tuple[str, ...]):
        identifier = str(identifier)
        if identifier in visited:
            return
        value = item(identifier)
        if not value or not visible(value):
            return
        visited.add(identifier)
        name = value.get("name")
        inner = value["inner"]
        if "use" in inner or "import" in inner:
            imported = inner.get("use", inner.get("import"))
            target = imported.get("id")
            if target is not None:
                visit(target, path)
            return
        if name == "internals" and not path:
            return
        if "module" in inner:
            module_path = path + (name,) if name and identifier != str(document["root"]) else path
            if module_path and module_path[0] == "internals":
                return
            for child in inner["module"]["items"]:
                child_item = item(child)
                if child_item and child_item.get("visibility") == "public":
                    visit(child, module_path)
            return
        if "function" in inner:
            if value.get("visibility") == "public":
                result.add(canonical_api("::".join(path + (name,))))
            return
        if "trait" in inner:
            for child in inner["trait"]["items"]:
                methods(child, name, trait=True)
            return
        for kind in ("struct", "enum", "union"):
            if kind not in inner:
                continue
            for implementation in inner[kind].get("impls", []):
                impl_item = item(implementation)
                if not impl_item or not visible(impl_item):
                    continue
                body = impl_item["inner"].get("impl")
                if body is None or body.get("trait") is not None or body.get("is_synthetic", False):
                    continue
                for child in body["items"]:
                    methods(child, name)
            return

    visit(document["root"], ())
    if not result:
        raise ValueError("rustdoc contains no reachable fgf-owned callables")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rustdoc", type=Path)
    arguments = parser.parse_args()
    document = json.loads(arguments.rustdoc.read_text())
    print(json.dumps(sorted(declared_callables(document)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
