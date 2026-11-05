// expect: mismatched types

// The raw dispatch entry takes a proof value whose type is private to the
// crate: outside code cannot name or construct it, so the call is rejected
// with a type error (E0308) at the proof argument. The entry's private
// visibility contributes a second diagnostic.
fn drive<F: fgf::FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    F::add_assign((), dst, src);
}
