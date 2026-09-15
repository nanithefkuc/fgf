//! Builds the Go adapter as a C archive and records what was linked.
//!
//! A competitor baseline that silently vanishes is worse than a missing one,
//! so an absent Go toolchain fails the build rather than cfg-disabling the
//! arm.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=go/main.go");
    println!("cargo::rerun-if-changed=go/go.mod");
    println!("cargo::rerun-if-changed=go/go.sum");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=PATH");

    let go = find_go();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir is set"));
    let archive = out_dir.join("libklauspost_adapter.a");

    let go_version = Command::new(&go)
        .arg("version")
        .output()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .map(|text| {
                    text.trim()
                        .strip_prefix("go version ")
                        .unwrap_or_else(|| text.trim())
                        .to_owned()
                })
                .map_err(std::io::Error::other)
        })
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", go.display()));
    let status = Command::new(&go)
        .env("CGO_ENABLED", "1")
        .current_dir(manifest.join("go"))
        .args([
            "build",
            "-trimpath",
            "-buildmode=c-archive",
            "-o",
            archive.to_str().expect("OUT_DIR is valid UTF-8"),
            ".",
        ])
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", go.display()));
    assert!(status.success(), "go build of the adapter failed");

    println!(
        "cargo::rustc-link-search=native={}",
        out_dir.to_str().expect("OUT_DIR is valid UTF-8")
    );
    println!("cargo::rustc-link-lib=static=klauspost_adapter");
    // The Go runtime's c-archive references these; modern glibc folds them
    // into libc, where the flags are accepted no-ops.
    println!("cargo::rustc-link-lib=dylib=pthread");
    println!("cargo::rustc-link-lib=dylib=dl");
    println!("cargo::rustc-link-lib=dylib=m");

    println!("cargo::rustc-env=KP_GO_VERSION={go_version}");
    println!(
        "cargo::rustc-env=KP_VERSION={}",
        pinned_version(&manifest.join("go/go.mod"))
    );
}

fn find_go() -> PathBuf {
    let path = env::var("PATH").unwrap_or_default();
    for directory in path.split(':') {
        if directory.is_empty() {
            continue;
        }
        let candidate = PathBuf::from(directory).join("go");
        if candidate.is_file() {
            return candidate;
        }
    }
    panic!(
        "the Go toolchain was not found in PATH.\n\
         Install Go (https://go.dev/dl/) and make `go` executable; the \
         klauspost/reedsolomon comparison cannot build without it."
    );
}

/// The reedsolomon requirement line in `go.mod` is the single source of the
/// pinned version; the binary cannot disagree with what it linked.
fn pinned_version(go_mod: &PathBuf) -> String {
    let text = fs::read_to_string(go_mod).expect("go/go.mod is readable");
    text.lines()
        .find(|line| line.starts_with("require github.com/klauspost/reedsolomon "))
        .and_then(|line| line.split_whitespace().last())
        .unwrap_or_else(|| panic!("no reedsolomon requirement found in {}", go_mod.display()))
        .to_owned()
}
