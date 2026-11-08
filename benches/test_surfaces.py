"""Synthetic protocol regressions; these fixtures are not benchmark measurements."""

import json
import unittest
from pathlib import Path

from check_surfaces import CheckError, SurfacesLog, group_of


def control(phase):
    return dict(case=f"control/xor/{phase}", api=["unchanged byte XOR control"],
                field="bytes", operation="control byte xor", scope="steady",
                bytes=64, lanes=64, sources=1, destinations=1,
                control=True, phase=phase)


def operation():
    return dict(case="bulk/mul-add/gf2p8-aes", api=["fgf::ops::mul_add"],
                field="gf2p8-aes", operation="mul-add", scope="steady",
                bytes=64, lanes=64, sources=1, destinations=1,
                control=False, phase=None)


def surface_log(*, duplicate=False, mode="check"):
    entries = [control("entry"), operation(), control("exit")]
    lines = ["fgf surfaces benchmark — backend: scalar"]
    if mode is not None:
        lines.append(f"BENCHMARK_MODE\t{mode}")
    lines += ["FIELD_BACKEND\tgf2p8-aes\tscalar",
             "FIELD_BACKEND\tprocess\tscalar",
             "HOST\tcpus_allowed_list=3\tsched_policy=5 idle"]
    lines.extend("SURFACE_INVENTORY\t" + json.dumps(entry) for entry in entries)
    if duplicate:
        lines.append("SURFACE_INVENTORY\t" + json.dumps(operation()))
    lines.extend(f"SURFACE_CHECKED\t{entry['case']}" for entry in entries)
    lines.append("SURFACES_CHECK_COMPLETE\tcases=3")
    return "\n".join(lines)


def native_entry(arm, schedule="fixed"):
    api = "fgf::ops::mul_add" if arm in {"fgf", "fgf-control"} else "foreign/region"
    return dict(case=f"gf16-presentations/gf16-rijndael16/mul_add/{schedule}/1x1/64/native/{arm}",
                api=[api], field="gf16-rijndael16", operation="mul_add",
                scope="native", bytes=64, lanes=32, sources=1, destinations=1,
                arm=arm, backend="scalar" if arm in {"fgf", "fgf-control"} else "native")


PRESENTATION_HEADER = ("GF16 presentations; fgf=fixture process_backend=Scalar; "
                       "rss=1.0 AVX2; rse=1.0 scalar GF16; "
                       "GF-Complete=1.03 fixture SSSE3 split-4/16 polynomial=0x1100b; "
                       f"Leopard-RS 1.x (pre-Leopard2)={'0' * 40} AVX2")
NATIVE_ARMS = ("fgf", "fgf-control", "gf-complete", "reed-solomon-erasure", "leopard")


def presentation_log(*, mode="check"):
    """One check-mode presentations section: a complete native mul_add group."""
    entries = [native_entry(arm) for arm in NATIVE_ARMS]
    lines = [PRESENTATION_HEADER]
    if mode is not None:
        lines.append(f"BENCHMARK_MODE\t{mode}")
    lines.extend("SURFACE_INVENTORY\t" + json.dumps(entry) for entry in entries)
    lines.append(f"PRESENTATIONS complete: 1 presentations, {len(entries)} "
                 "measurements validated without timing.")
    return "\n".join(lines)


class SurfaceProtocolTests(unittest.TestCase):
    def test_valid_check_captures_validate(self):
        surfaces = SurfacesLog(Path("surfaces-check.log"))
        surfaces.parse(surface_log())
        surfaces.validate_check_capture()
        presentations = SurfacesLog(Path("presentations-check.log"))
        presentations.parse(presentation_log())
        presentations.validate_check_capture()

    def test_duplicate_inventory_does_not_collapse_into_a_valid_identity(self):
        parsed = SurfacesLog(Path("duplicate.log"))
        parsed.parse(surface_log(duplicate=True))
        with self.assertRaises(CheckError):
            parsed.validate_section("surfaces")

    def test_native_group_matches_operation_not_foreign_api_or_host_backend(self):
        fgf = native_entry("fgf")
        foreign = native_entry("gf-complete")
        other_host = dict(fgf, backend="v4x")
        self.assertEqual(group_of(fgf), group_of(foreign))
        self.assertEqual(group_of(fgf), group_of(other_host))
        self.assertNotEqual(group_of(fgf), group_of(native_entry("fgf", "cycling16")))

    def test_missing_applicable_native_arm_is_rejected(self):
        entries = [native_entry(arm) for arm in
                   ("fgf", "fgf-control", "gf-complete", "reed-solomon-erasure")]
        lines = [PRESENTATION_HEADER, "BENCHMARK_MODE\tcheck"]
        lines.extend("SURFACE_INVENTORY\t" + json.dumps(entry) for entry in entries)
        lines.append("PRESENTATIONS complete: 1 presentations, 4 measurements validated without timing.")
        parsed = SurfacesLog(Path("missing-native-arm.log"))
        parsed.parse("\n".join(lines))
        with self.assertRaises(CheckError):
            parsed.validate_check_capture()

    def test_check_logs_require_a_mode_marker(self):
        with self.assertRaises(CheckError):
            SurfacesLog(Path("modeless.log")).parse(surface_log(mode=None))
        with self.assertRaises(CheckError):
            SurfacesLog(Path("modeless-presentations.log")).parse(
                presentation_log(mode=None))

    def test_check_capture_requires_check_mode(self):
        parsed = SurfacesLog(Path("full-check.log"))
        parsed.parse(surface_log(mode="full"))
        with self.assertRaises(CheckError):
            parsed.validate_check_capture()
        full_presentations = SurfacesLog(Path("full-presentations.log"))
        full_presentations.parse(presentation_log(mode="full"))
        with self.assertRaises(CheckError):
            full_presentations.validate_check_capture()


    def test_process_record_with_capability_flags_fails_closed(self):
        legacy = surface_log().replace(
            "FIELD_BACKEND\tprocess\tscalar",
            "FIELD_BACKEND\tprocess\tscalar\tnative_mul=false\tblocked_rows=false")
        parsed = SurfacesLog(Path("legacy-process-record.log"))
        with self.assertRaises(CheckError):
            parsed.parse(legacy)

    def test_selected_check_capture_parses_but_fails_full_check_validation(self):
        entries = [control("entry"), operation(), control("exit")]
        selected = surface_log().replace(
            "SURFACES_CHECK_COMPLETE\tcases=3",
            "SURFACES_SELECTED_CHECK_COMPLETE\tcases=3").replace(
            "\n".join(f"SURFACE_CHECKED\t{entry['case']}" for entry in entries),
            "\n".join(f"SURFACE_CHECKED\t{entry['case']}" for entry in entries[:1]))
        parsed = SurfacesLog(Path("selected-check.log"))
        parsed.parse(selected)
        with self.assertRaises(CheckError) as raised:
            parsed.validate_check_capture()
        self.assertIn("selected", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
