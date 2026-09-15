//! Locates the system Intel ISA-L and records the version that was linked.
//!
//! A competitor baseline that silently vanishes is worse than a missing one,
//! so an absent library fails the build rather than cfg-disabling the arm.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    match pkg_config::Config::new()
        .atleast_version("2.30")
        .probe("libisal")
    {
        Ok(library) => println!("cargo::rustc-env=ISAL_VERSION={}", library.version),
        Err(error) => panic!(
            "ISA-L not found through pkg-config: {error}\n\
             Install Intel ISA-L 2.30 or newer (Arch: `isa-l`, Debian: `libisal-dev`, \
             or build https://github.com/intel/isa-l) and make `libisal.pc` visible \
             through PKG_CONFIG_PATH."
        ),
    }
}
