// expect-code: E0624
use fgf::FieldKernels;

// Bound elaboration names the raw dispatch entry, but its proof argument is
// private to the crate: outside code cannot construct it, so the call site
// cannot compile.
fn drive<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    F::add_assign((), dst, src);
}
