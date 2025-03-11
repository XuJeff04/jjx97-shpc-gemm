
/* Computes C = AB + C */
void shpc_dgemm(int m, int n, int k,
                double *A, int rsA, int csA,
                double *B, int rsB, int csB,
                double *C, int rsC, int csC);

inline void ukernel(double *A, int rsA, int csA, double *B, int rsB, int csB, double *C, int rsC, int csC);


#define ITERATION(p) \
    alpha_0123_p = _mm256_loadu_pd(&A[0*rsA + (p)*csA]); \
    alpha_4567_p = _mm256_loadu_pd(&A[4*rsA + (p)*csA]); \
    beta_p_0 = _mm256_broadcast_sd(&B[(p)*rsB + 0*csB]); \
    beta_p_1 = _mm256_broadcast_sd(&B[(p)*rsB + 1*csB]); \
    beta_p_2 = _mm256_broadcast_sd(&B[(p)*rsB + 2*csB]); \
    beta_p_3 = _mm256_broadcast_sd(&B[(p)*rsB + 3*csB]); \
    gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p_0, gamma_0123_0); \
    gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p_1, gamma_0123_1); \
    gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p_2, gamma_0123_2); \
    gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p_3, gamma_0123_3); \
    gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p_0, gamma_4567_0); \
    gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p_1, gamma_4567_1); \
    gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p_2, gamma_4567_2); \
    gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p_3, gamma_4567_3);

// Generate 8 iterations
#define UNROLL_8(start) \
    ITERATION(start) \
    ITERATION(start+1) \
    ITERATION(start+2) \
    ITERATION(start+3) \
    ITERATION(start+4) \
    ITERATION(start+5) \
    ITERATION(start+6) \
    ITERATION(start+7)

// Generate 32 iterations
#define UNROLL_32(start) \
    UNROLL_8(start) \
    UNROLL_8(start+8) \
    UNROLL_8(start+16) \
    UNROLL_8(start+24)

// Generate 256 iterations using 8 blocks of 32 iterations
#define UNROLL_256 \
    __m256d alpha_0123_p; \
    __m256d alpha_4567_p; \
    __m256d beta_p_0; \
    __m256d beta_p_1; \
    __m256d beta_p_2; \
    __m256d beta_p_3; \
    UNROLL_32(0) \
    UNROLL_32(32) \
    UNROLL_32(64) \
    UNROLL_32(96) \
    UNROLL_32(128) \
    UNROLL_32(160) \
    UNROLL_32(192) \
    UNROLL_32(224)