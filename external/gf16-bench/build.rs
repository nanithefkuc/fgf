//! Builds pinned permissive competitors and the benchmark-only native bridges.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const LEOPARD: &str = "6e5725ebdf9da4370b0bcc4f70fa8eb66f4e6198";
const COMPLETE: &str = "a6862d10c9db467148f20eef2c6445ac9afd94d8";

fn run(command: &mut Command) {
    let output = command.output().expect("execute native source checkout");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn checkout(root: &Path, name: &str, repository: &str, revision: &str) -> PathBuf {
    let path = root.join(name);
    if !path.exists() {
        run(Command::new("git")
            .args(["clone", "--filter=blob:none", "--no-checkout", repository])
            .arg(&path));
        run(Command::new("git")
            .arg("-C")
            .arg(&path)
            .args(["fetch", "--depth", "1", "origin", revision]));
        run(Command::new("git")
            .arg("-C")
            .arg(&path)
            .args(["checkout", "--detach", revision]));
    }
    let output = Command::new("git")
        .arg("-C")
        .arg(&path)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("read pinned revision");
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        revision,
        "native checkout must retain its pin"
    );
    path
}

fn main() {
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
        "x86_64",
        "the native competitor campaign requires x86-64 AVX2 hardware"
    );
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=native/leopard.cpp");
    println!("cargo::rerun-if-changed=native/complete.c");
    println!("cargo::rerun-if-changed=../../Cargo.toml");
    let manifest = std::fs::read_to_string("../../Cargo.toml").unwrap();
    let fgf_version = manifest
        .lines()
        .find_map(|line| line.strip_prefix("version = \""))
        .and_then(|rest| rest.strip_suffix('"'))
        .expect("the fgf manifest must declare its package version");
    println!("cargo::rustc-env=FGF_VERSION={fgf_version}");
    let root = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let leopard = checkout(
        &root,
        "leopard",
        "https://github.com/catid/leopard.git",
        LEOPARD,
    );
    let complete = checkout(
        &root,
        "gf-complete",
        "https://github.com/ceph/gf-complete.git",
        COMPLETE,
    );
    cc::Build::new()
        .cpp(true)
        .std("c++11")
        .opt_level(3)
        .warnings(false)
        .flag("-mavx2")
        .include(&leopard)
        .file("native/leopard.cpp")
        .file(leopard.join("LeopardCommon.cpp"))
        .compile("gf16_leopard");
    let mut build = cc::Build::new();
    build
        .std("gnu11")
        .opt_level(3)
        .warnings(false)
        .include(complete.join("include"))
        .include(complete.join("src"))
        .flag("-mssse3")
        .flag("-msse4.2")
        .flag("-mpclmul")
        .define("INTEL_SSE2", None)
        .define("INTEL_SSE3", None)
        .define("INTEL_SSSE3", None)
        .define("INTEL_SSE4", None)
        .define("INTEL_SSE4_PCLMUL", None)
        .file("native/complete.c");
    for source in [
        "gf",
        "gf_wgen",
        "gf_w4",
        "gf_w8",
        "gf_w32",
        "gf_w64",
        "gf_w128",
        "gf_rand",
        "gf_general",
        "gf_cpu",
        "gf_method",
    ] {
        build.file(complete.join(format!("src/{source}.c")));
    }
    build.compile("gf16_complete");
    println!("cargo::rustc-env=LEOPARD_REVISION={LEOPARD}");
    println!("cargo::rustc-env=COMPLETE_REVISION={COMPLETE}");
}
