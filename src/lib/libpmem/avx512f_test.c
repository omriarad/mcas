#include <immintrin.h>
#include <stdint.h>
int main(){ uint64_t v[8]; __m512i zmm0 = _mm512_loadu_si512((__m512i *)&v); return 0;}

