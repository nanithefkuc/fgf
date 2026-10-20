"""Synthetic protocol regressions; these fixtures are not benchmark measurements."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from check_surfaces import SurfacesLog, group_of, workload_of
from publish import PublishError, Renderer
from surface_publish import SurfaceEntry, _surface_scope_sections


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


def surface_log(*, duplicate=False, host=True, backend="scalar", mode="full"):
    entries = [control("entry"), operation(), control("exit")]
    lines = ["fgf surfaces benchmark — backend: scalar"]
    if mode is not None:
        lines.append(f"BENCHMARK_MODE\t{mode}")
    lines += ["FIELD_BACKEND\tgf2p8-aes\tscalar",
             "FIELD_BACKEND\tprocess\tscalar\tnative_mul=false\tblocked_rows=false"]
    if host:
        lines.append("HOST\tcpus_allowed_list=3\tsched_policy=5 idle")
    lines.extend("SURFACE_INVENTORY\t" + json.dumps(entry) for entry in entries)
    lines.append('TIMER_FLOOR\t{"ns":0.5,"samples":24}')
    for entry in entries:
        record = dict(entry, backend=backend if not entry["control"] else "scalar",
                      ns=10.0, samples=24, p10_ns=9.0, p90_ns=11.0)
        line = "SURFACE\t" + json.dumps(record)
        lines.append(line)
        if duplicate and not entry["control"]:
            lines.append(line)
    lines.append("SURFACES\tcases=3\telapsed_s=1.000")
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


def timed_presentation_log(*, mode="full", schedule="fixed", fgf_reps=8,
                           control_reps=8, native_reps=None, omit_reps=False,
                           samples=40, control_samples=None):
    """One timed presentations section: a complete native mul_add group whose
    fgf and control arms may carry deliberately mismatched batch sizes."""
    entries = [native_entry(arm, schedule) for arm in NATIVE_ARMS]
    lines = [PRESENTATION_HEADER]
    if mode is not None:
        lines.append(f"BENCHMARK_MODE\t{mode}")
    lines.extend("SURFACE_INVENTORY\t" + json.dumps(entry) for entry in entries)
    lines.append("HOST\tcpus_allowed_list=3\tsched_policy=5 idle")
    for entry in entries:
        count = (control_samples if entry["arm"] == "fgf-control"
                 and control_samples is not None else samples)
        record = dict(entry, ns=10.0, samples=count, p10_ns=9.0, p90_ns=11.0)
        if not omit_reps:
            if entry["arm"] == "fgf-control":
                record["batch_reps"] = control_reps
            elif entry["arm"] == "fgf":
                record["batch_reps"] = fgf_reps
            else:
                record["batch_reps"] = fgf_reps if native_reps is None else native_reps
        lines.append("GF16_SURFACE\t" + json.dumps(record))
    lines.append(f"PRESENTATIONS complete: {len(entries)} presentations, "
                 f"{len(entries)} measurements.")
    return "\n".join(lines)


class SurfaceProtocolTests(unittest.TestCase):
    def test_duplicate_timing_does_not_collapse_into_a_valid_inventory(self):
        parsed = SurfacesLog(Path("duplicate.log"))
        parsed.parse(surface_log(duplicate=True))
        with self.assertRaises(PublishError):
            parsed.validate_section("surfaces")

    def test_timing_without_host_placement_is_rejected(self):
        parsed = SurfacesLog(Path("missing-host.log"))
        parsed.parse(surface_log(host=False))
        with self.assertRaises(PublishError):
            parsed.validate_section("surfaces")

    def test_record_backend_must_match_the_field_route(self):
        parsed = SurfacesLog(Path("wrong-backend.log"))
        parsed.parse(surface_log(backend="v4x"))
        with self.assertRaises(PublishError):
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
        with self.assertRaises(PublishError):
            parsed.validate_check_capture()

    def test_smoke_rounds_parse_but_are_rejected_as_campaign(self):
        parsed = SurfacesLog(Path("smoke.log"))
        parsed.parse(surface_log(mode="smoke"))
        parsed.validate_section("surfaces")
        with self.assertRaises(PublishError):
            parsed.require_full()
        smoke_presentations = SurfacesLog(Path("smoke-presentations.log"))
        smoke_presentations.parse(timed_presentation_log(mode="smoke"))
        with self.assertRaises(PublishError):
            smoke_presentations.require_full()

    def test_round_logs_require_a_mode_marker(self):
        parsed = SurfacesLog(Path("modeless.log"))
        with self.assertRaises(PublishError):
            parsed.parse(surface_log(mode=None))
        with self.assertRaises(PublishError):
            SurfacesLog(Path("modeless-presentations.log")).parse(
                timed_presentation_log(mode=None))

    def test_check_capture_requires_check_mode(self):
        parsed = SurfacesLog(Path("full-check.log"))
        parsed.parse(surface_log(mode="full"))
        with self.assertRaises(PublishError):
            parsed.validate_check_capture()

    def test_timed_presentations_require_positive_integer_batch_reps(self):
        with self.assertRaises(PublishError):
            SurfacesLog(Path("no-reps.log")).parse(timed_presentation_log(omit_reps=True))
        for reps in (0, -8, 2.5, "8", True):
            with self.assertRaises(PublishError):
                SurfacesLog(Path(f"reps-{reps!r}.log")).parse(
                    timed_presentation_log(fgf_reps=reps))

    def test_native_fgf_and_control_arms_share_one_batch_size(self):
        parsed = SurfacesLog(Path("mismatched-batches.log"))
        parsed.parse(timed_presentation_log(fgf_reps=8, control_reps=16))
        with self.assertRaises(PublishError):
            parsed.validate_section("gf16-presentations")

    def test_cycling16_batches_must_hold_whole_coefficient_cycles(self):
        parsed = SurfacesLog(Path("partial-cycle.log"))
        parsed.parse(timed_presentation_log(schedule="cycling16", fgf_reps=16,
                                            control_reps=16, native_reps=17))
        with self.assertRaises(PublishError):
            parsed.validate_section("gf16-presentations")
        whole = SurfacesLog(Path("whole-cycles.log"))
        whole.parse(timed_presentation_log(schedule="cycling16", fgf_reps=16,
                                           control_reps=16, native_reps=32))
        whole.validate_section("gf16-presentations")
        whole.require_full()

    def test_native_arms_must_share_one_balanced_sample_count(self):
        parsed = SurfacesLog(Path("unequal-samples.log"))
        parsed.parse(timed_presentation_log(samples=40, control_samples=30))
        with self.assertRaises(PublishError):
            parsed.validate_section("gf16-presentations")

    def test_native_samples_must_cover_whole_balanced_cycles(self):
        # Five arms balance over ten orders; 35 samples stops mid-cycle.
        parsed = SurfacesLog(Path("partial-balance.log"))
        parsed.parse(timed_presentation_log(samples=35))
        with self.assertRaises(PublishError):
            parsed.validate_section("gf16-presentations")

    def test_full_native_runs_keep_the_sample_floor_but_smoke_does_not(self):
        short = SurfacesLog(Path("short-full.log"))
        short.parse(timed_presentation_log(samples=20))
        with self.assertRaises(PublishError):
            short.validate_section("gf16-presentations")
        smoke = SurfacesLog(Path("short-smoke.log"))
        smoke.parse(timed_presentation_log(mode="smoke", samples=10))
        smoke.validate_section("gf16-presentations")


class SurfacePublicationTests(unittest.TestCase):
    def test_distinct_primality_workloads_have_distinct_evidence_addresses(self):
        # These inputs share a field and recipe, but their timings describe
        # different work. A consumer must be able to locate each cell uniquely.
        entries = []
        aggregates = {}
        for index, operation in enumerate(("even-composite", "large-prime", "semiprime")):
            first = dict(case=f"convert/is-prime/{operation}", field="u64",
                         operation=operation, scope="steady", api=["fgf::field::is_prime"],
                         bytes=0, lanes=1, sources=1, destinations=1,
                         control=False, phase=None)
            entry = SurfaceEntry(first)
            for host in ("one", "two"):
                entry.add(host, 1, dict(ns=10.0 + index, backend="scalar"), "fixture")
                aggregates[(workload_of(first, "surfaces"), host)] = 10.0 + index
            entries.append(entry)
        campaign = SimpleNamespace(
            hosts=["one", "two"], available_hosts={"one", "two"},
            surface_aggregates=aggregates,
            surface_logs=[dict(host=host, round=1, timer_floor={"ns": 0.5})
                          for host in ("one", "two")])
        renderer = Renderer(campaign)
        _surface_scope_sections(campaign, renderer, "v3/public-api/encodings.md", entries)
        addresses = [(cell["page"], cell["table"], cell["row"], cell["column"])
                     for cell in renderer.cells.refs]
        self.assertEqual(len(addresses), 3)
        self.assertEqual(len(set(addresses)), 3)


if __name__ == "__main__":
    unittest.main()
