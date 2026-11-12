//! Checked calls into the pinned native GF16 competitors.

use archmage::SimdToken;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

// SAFETY:
// ABI
// SINCE: the x86-64 native bridge definitions use the C ABI with unsigned/int/u64 scalars and opaque or byte pointers matching these declarations.
// THUS: the declarations match the linked functions' calling convention and representations.
#[allow(unsafe_code)]
unsafe extern "C" {
    fn gf16_complete_new() -> *mut c_void;
    fn gf16_complete_free(context: *mut c_void);
    fn gf16_complete_mul(context: *mut c_void, a: u32, b: u32) -> u32;
    fn gf16_complete_region(
        context: *mut c_void,
        dst: *mut u8,
        src: *const u8,
        coefficient: u32,
        bytes: i32,
        add: i32,
    );
    fn gf16_leopard_init() -> i32;
    fn gf16_leopard_mul(a: u32, b: u32) -> u32;
    fn gf16_leopard_region(
        dst: *mut u8,
        src: *const u8,
        coefficient_log: u32,
        bytes: u64,
        add: i32,
    );
}

pub struct Native {
    complete: NonNull<c_void>,
    _token: archmage::X64V3GfniCryptoToken,
    _single_thread: PhantomData<Rc<()>>,
}

impl Native {
    #[allow(unsafe_code)]
    pub fn new() -> Self {
        let token = archmage::X64V3GfniCryptoToken::summon()
            .expect("native competitor build requires AVX2, GFNI and crypto hardware");
        // SAFETY:
        // CPU FEATURES
        // SINCE: `token` proves the AVX2, SSSE3, SSE4 and PCLMUL instructions used by both native builds.
        // THUS: initialization can legally execute their compiled instructions.
        // OWNERSHIP
        // SINCE: the C constructor allocates a fresh GF-Complete context; Leopard initializes process-global tables without callbacks.
        // THUS: no borrowed Rust storage escapes or conflicts with native initialization.
        let complete = unsafe {
            assert_eq!(
                gf16_leopard_init(),
                1,
                "Leopard must initialize its AVX2 arm"
            );
            NonNull::new(gf16_complete_new()).expect(
                "GF-Complete must select polynomial 0x1100b and its SSSE3 split-table region arm",
            )
        };
        Self {
            complete,
            _token: token,
            _single_thread: PhantomData,
        }
    }

    #[allow(unsafe_code)]
    pub fn complete_mul(&self, a: u16, b: u16) -> u16 {
        // SAFETY:
        // LIFETIME
        // SINCE: `self` owns the live native context until Drop, and u16 arguments are valid GF16 values.
        // THUS: the scalar call reads a live initialized field context and in-range operands.
        // CPU FEATURES
        // SINCE: `self._token` proves every instruction enabled in the native build.
        // THUS: its scalar field implementation can execute legally.
        unsafe { gf16_complete_mul(self.complete.as_ptr(), u32::from(a), u32::from(b)) as u16 }
    }

    #[allow(unsafe_code)]
    pub fn leopard_mul(&self, a: u16, b: u16) -> u16 {
        // SAFETY:
        // INITIALIZATION
        // SINCE: Native::new initializes Leopard's tables before constructing self, and both operands index at most 65535.
        // THUS: the scalar helper reads initialized, in-bounds tables.
        // CPU FEATURES
        // SINCE: `self._token` proves the native build's instruction set.
        // THUS: the scalar helper executes legally.
        unsafe { gf16_leopard_mul(u32::from(a), u32::from(b)) as u16 }
    }

    #[allow(unsafe_code)]
    pub fn complete_region(&self, dst: &mut [u8], src: &[u8], coefficient: u16, add: bool) {
        check_complete(dst, src.len());
        assert_eq!(src.as_ptr().align_offset(16), 0);
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: equal-length initialized slices contain complete words; check_complete bounds the signed byte count.
        // THUS: the native region reads and writes only live slice storage and a live owned context.
        // ALIGNMENT
        // SINCE: both checked pointers are 16-byte aligned and their lengths are multiples of 64.
        // THUS: the native SIMD body and uint16 accesses meet their alignment requirements.
        // ALIASING
        // SINCE: the exclusive destination borrow is disjoint from the shared source borrow; Native is not Send or Sync.
        // THUS: destination writes and native scratch updates conflict with no other access.
        // CPU FEATURES
        // SINCE: `self._token` proves the enabled native instruction set.
        // THUS: the selected SSSE3 implementation can execute legally.
        unsafe {
            gf16_complete_region(
                self.complete.as_ptr(),
                dst.as_mut_ptr(),
                src.as_ptr(),
                u32::from(coefficient),
                dst.len() as i32,
                i32::from(add),
            )
        };
    }

    #[allow(unsafe_code)]
    pub fn complete_assign(&self, dst: &mut [u8], coefficient: u16) {
        check_complete(dst, dst.len());
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: the checked initialized slice bounds the signed byte count and the context is owned by self.
        // THUS: every read and write stays inside live storage.
        // ALIGNMENT
        // SINCE: check_complete establishes 16-byte alignment and whole SIMD tiles.
        // THUS: native vector and uint16 accesses are aligned.
        // ALIASING
        // SINCE: GF-Complete's region body loads each source tile before storing the corresponding destination tile; both pointers derive from one exclusive borrow.
        // THUS: exact in-place execution is valid and does not conflict with another reference.
        // CPU FEATURES
        // SINCE: `self._token` proves the enabled native instruction set.
        // THUS: the selected SSSE3 implementation can execute legally.
        unsafe {
            gf16_complete_region(
                self.complete.as_ptr(),
                dst.as_mut_ptr(),
                dst.as_ptr(),
                u32::from(coefficient),
                dst.len() as i32,
                0,
            )
        };
    }

    #[allow(unsafe_code)]
    pub fn leopard_region(&self, dst: &mut [u8], src: &[u8], coefficient_log: u16, add: bool) {
        assert_eq!(dst.len(), src.len());
        assert!(!dst.is_empty() && dst.len().is_multiple_of(64));
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: initialized equal-length slices contain a positive whole number of ALTMAP tiles; coefficient_log is a u16 table index.
        // THUS: all unaligned SIMD accesses and table reads stay in bounds.
        // ALIASING
        // SINCE: exclusive destination and shared source borrows are disjoint.
        // THUS: the upstream restrict-qualified overwrite and accumulating macro have no overlapping buffers.
        // INITIALIZATION
        // SINCE: Native::new initializes Leopard's multiplication tables before constructing self.
        // THUS: the native calls read fully initialized table entries.
        // CPU FEATURES
        // SINCE: `self._token` proves AVX2 and construction verifies Leopard selected AVX2.
        // THUS: the native AVX2 region can execute legally.
        unsafe {
            gf16_leopard_region(
                dst.as_mut_ptr(),
                src.as_ptr(),
                u32::from(coefficient_log),
                dst.len() as u64,
                i32::from(add),
            )
        };
    }
}

#[allow(unsafe_code)]
impl Drop for Native {
    fn drop(&mut self) {
        // SAFETY:
        // OWNERSHIP
        // SINCE: self uniquely owns the pointer returned by the native constructor; no reference to it escapes Native.
        // THUS: the native destructor frees it exactly once after its last access.
        // CPU FEATURES
        // SINCE: self retains the native build's capability token through this call.
        // THUS: the native destructor's compiled instructions can execute legally.
        unsafe { gf16_complete_free(self.complete.as_ptr()) };
    }
}

fn check_complete(dst: &[u8], source_len: usize) {
    assert_eq!(dst.len(), source_len);
    assert!(!dst.is_empty() && dst.len().is_multiple_of(64));
    assert!(i32::try_from(dst.len()).is_ok());
    assert_eq!(dst.as_ptr().align_offset(16), 0);
}
