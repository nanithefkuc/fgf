// expect-rejection
// `FieldBuffer` is sealed: the crate's rows define every byte layout, so a
// downstream type cannot implement it.
#[derive(Clone, Copy, Debug)]
pub struct Downstream;

impl fgf::FieldBuffer for Downstream {
    const BYTES: usize = 1;
    const STORAGE_BITS: u32 = 8;
    fn decode(_bytes: &[u8]) -> fgf::Elem<Self> {
        fgf::Elem::<Self>::ZERO
    }
    fn encode(_bytes: &mut [u8], _value: fgf::Elem<Self>) {}
}
