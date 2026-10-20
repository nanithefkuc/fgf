// expect-rejection
use fgf::FieldKernels;

// Bound elaboration names the raw dispatch entry, but its proof is private
// to the crate: outside code can neither name the type nor construct the
// value, so the proof argument cannot be supplied and the call site cannot
// compile.
fn drive<F: FieldKernels>(dst: &mut [u8], src: &[u8]) {
    let proof: fgf::kernel::RawDispatch = fgf::kernel::RawDispatch;
    F::add_assign(proof, dst, src);
}
