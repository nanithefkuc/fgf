/* Native GF-Complete region calls; buffers use little-endian polynomial words. */
#include "gf_complete.h"
#include "gf_int.h"
#include <stdlib.h>

#include "gf_w16.c"

void *gf16_complete_new(void) {
    gf_t *gf = calloc(1, sizeof(*gf));
    if (!gf) return NULL;
    if (!gf_init_easy(gf, 16)) { free(gf); return NULL; }
    if (((gf_internal_t *)gf->scratch)->prim_poly != 0x1100b ||
        gf->multiply_region.w32 != gf_w16_split_4_16_lazy_sse_multiply_region) {
        gf_free(gf, 1); free(gf); return NULL;
    }
    return gf;
}

void gf16_complete_free(void *context) {
    gf_t *gf = context;
    gf_free(gf, 1);
    free(gf);
}

unsigned gf16_complete_mul(void *context, unsigned a, unsigned b) {
    gf_t *gf = context;
    return gf->multiply.w32(gf, a, b);
}

void gf16_complete_region(void *context, unsigned char *dst,
                          const unsigned char *src, unsigned coefficient,
                          int bytes, int add) {
    gf_t *gf = context;
    gf->multiply_region.w32(gf, (void *)src, dst, coefficient, bytes, add);
}
