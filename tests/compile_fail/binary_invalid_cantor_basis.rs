// expect: basis map is not bijective
use fgf::binary::Cantor;
use fgf::{Elem, FiniteField};

type Row = FiniteField<2, 8, Cantor<0x11B, 0>>;

const _: Elem<Row> = Elem::<Row>::ONE;
