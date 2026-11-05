// expect: polynomial is reducible
use fgf::binary::Polynomial;
use fgf::{Elem, FiniteField};

type Row = FiniteField<2, 8, Polynomial<0x101>>;

const _: Elem<Row> = Elem::<Row>::ONE;
