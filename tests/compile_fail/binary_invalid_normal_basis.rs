// expect: basis map is not bijective
use fgf::binary::Normal;
use fgf::{Elem, FiniteField};

type Row = FiniteField<2, 8, Normal<0x11B, 1>>;

const _: Elem<Row> = Elem::<Row>::ONE;
