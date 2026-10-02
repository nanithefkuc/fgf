// Native Leopard-RS 1.x (pre-Leopard2) multiplication; ALTMAP tiles hold low bytes then high bytes.
#include "LeopardFF16.cpp"

extern "C" int gf16_leopard_init() {
    leopard::InitializeCPUArch();
    return leopard::CpuHasAVX2 && leopard::ff16::Initialize();
}

extern "C" unsigned gf16_leopard_mul(unsigned a, unsigned b) {
    if (!a || !b) return 0;
    return leopard::ff16::MultiplyLog(static_cast<uint16_t>(a), leopard::ff16::LogLUT[b]);
}

extern "C" void gf16_leopard_region(unsigned char *dst, const unsigned char *src,
                                     unsigned coefficient_log, unsigned long long bytes,
                                     int add) {
    using namespace leopard::ff16;
    if (!add) {
        mul_mem(dst, src, static_cast<uint16_t>(coefficient_log), bytes);
        return;
    }
    LEO_MUL_TABLES_256(0, coefficient_log);
    const LEO_M256 clr_mask = _mm256_set1_epi8(0x0f);
    for (unsigned long long offset = 0; offset < bytes; offset += 64) {
        LEO_M256 x_lo = _mm256_loadu_si256(reinterpret_cast<const LEO_M256 *>(dst + offset));
        LEO_M256 x_hi = _mm256_loadu_si256(reinterpret_cast<const LEO_M256 *>(dst + offset + 32));
        const LEO_M256 y_lo = _mm256_loadu_si256(reinterpret_cast<const LEO_M256 *>(src + offset));
        const LEO_M256 y_hi = _mm256_loadu_si256(reinterpret_cast<const LEO_M256 *>(src + offset + 32));
        LEO_MULADD_256(x_lo, x_hi, y_lo, y_hi, 0);
        _mm256_storeu_si256(reinterpret_cast<LEO_M256 *>(dst + offset), x_lo);
        _mm256_storeu_si256(reinterpret_cast<LEO_M256 *>(dst + offset + 32), x_hi);
    }
}
