// expect-rejection
use fgf::FieldBuffer;

fn drive<F: FieldBuffer>(dst: &mut [u8], src: &[u8]) {
    F::sub_assign_bytes(dst, src);
}
