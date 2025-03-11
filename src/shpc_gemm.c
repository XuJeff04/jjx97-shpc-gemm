#include "assignment3.h"
#include "immintrin.h"
//all sizes in units of doubles
#define R_SIZE 4 //256 bits, 4 total doubles mr and nr
#define MAIN_MEMORY_SIZE (1UL << 32)
#define L3_CACHE_SIZE (1U << 21)
#define L2_CACHE_SIZE (1U << 15) // private L2 Caches, 8 cores
#define L1_CACHE_SIZE (1U << 12)
#define _MR 8
#define _NR 4
#define _NC 1024
#define _KC 256
#define _MC 96
#define ALIGN(size, alignment) (((size) + (alignment-1)) & ~(alignment-1))


static void padMat(double* newMat, double* oldMat, int m, int n, int rsC, int csC);
static void unpadMat(double* newMat, double* oldMat, int m, int n, int rsC, int csC);

double Apacked[_MC * _KC] __attribute__((aligned(4096)));
double Bpacked[_KC * _NC] __attribute__((aligned(4096)));

void shpc_dgemm( int m, int n, int k,
                    double *A, int rsA, int csA,                                
                    double *B, int rsB, int csB,                                
                    double *C, int rsC, int csC )
{
    double* newC;
    if (posix_memalign((void**)&newC, 4096, ALIGN(m, _MR) * ALIGN(n, _NR) * sizeof(double))) {
        return;
    }
    memset(newC, 0, ALIGN(m, _MR) * ALIGN(n, _NR) * sizeof(double));
    padMat(newC, C, m, n, rsC, csC);
    unsigned newCsC = ALIGN(m, _MR);
    for (int j = 0; j < n; j+=_NC) {
        double* Cj = newC+(newCsC * j);
        double* Bj = B+(csB * j);
        for (int p = 0; p < k; p += _KC) {
            //pack loops:
            for (int j_pack = 0; j_pack < _NC; j_pack+=_NR) {
                for (int _j_pack = 0; _j_pack < _NR; _j_pack++) {
                    for (int i_pack = 0; i_pack < _KC; i_pack++) {
                        if (i_pack+p >= k || _j_pack+j_pack+j >= n) {
                            Bpacked[i_pack * _NR + j_pack*_KC + _j_pack] = 0.0;
                        } else {
                            Bpacked[i_pack * _NR + j_pack*_KC + _j_pack] = *(Bj+((i_pack+p)*rsB)+((j_pack + _j_pack)*csB));
                        }
                    }
                }
            }
            //next loop:
            for (int i = 0; i < m; i+=_MC) {
                //pack loops:
                for (int j_pack = 0; j_pack < _KC; j_pack++) {
                    for (int i_pack = 0; i_pack < _MC; i_pack+=_MR) {
                        for (int _i_pack = 0; _i_pack < _MR; _i_pack++) {
                            if (i_pack + _i_pack + i > m || j_pack+p > k) {
                                Apacked[i_pack * _KC + j_pack*_MR + _i_pack] = 0.0;
                            } else {
                                Apacked[i_pack * _KC + j_pack*_MR + _i_pack] = *(A+((i_pack+i+_i_pack)*rsA)+((j_pack+p)*csA));
                            }
                        }
                    }
                }
                //next loop:
                for (int _j = 0; _j < _NC && j +_j < ALIGN(n, _NR); _j+=_NR) {
                    for (int _i = 0; _i < _MC && i +_i < ALIGN(m, _MR); _i+=_MR) {
                        ukernel(&Apacked[_i*_KC], 1, _MR, &Bpacked[_j*_KC], _NR, 1, Cj+(i+_i)+_j*newCsC, 1, newCsC);
                    }
                }
            }
        }
    }
    unpadMat(newC, C, m, n, rsC, csC);
    free(newC);
}

inline void ukernel (double *A, int rsA, int csA, double* B, int rsB, int csB, double* C, int rsC, int csC) {
    __m256d gamma_0123_0 = _mm256_loadu_pd(&C[0 * rsC + 0 * csC]);
    __m256d gamma_0123_1 = _mm256_loadu_pd(&C[0 * rsC + 1 * csC]);
    __m256d gamma_0123_2 = _mm256_loadu_pd(&C[0 * rsC + 2 * csC]);
    __m256d gamma_0123_3 = _mm256_loadu_pd(&C[0 * rsC + 3 * csC]);
    __m256d gamma_4567_0 = _mm256_loadu_pd(&C[4 * rsC + 0 * csC]);
    __m256d gamma_4567_1 = _mm256_loadu_pd(&C[4 * rsC + 1 * csC]);
    __m256d gamma_4567_2 = _mm256_loadu_pd(&C[4 * rsC + 2 * csC]);
    __m256d gamma_4567_3 = _mm256_loadu_pd(&C[4 * rsC + 3 * csC]);

    // Matrix multiplication loop
    for (int p = 0; p < _KC; p++) {
        __m256d alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + p * csA]);
        __m256d alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + p * csA]);

        __m256d beta_p_0 = _mm256_broadcast_sd(&B[p * rsB + 0 * csB]);
        __m256d beta_p_1 = _mm256_broadcast_sd(&B[p * rsB + 1 * csB]);
        __m256d beta_p_2 = _mm256_broadcast_sd(&B[p * rsB + 2 * csB]);
        __m256d beta_p_3 = _mm256_broadcast_sd(&B[p * rsB + 3 * csB]);

        gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p_0, gamma_0123_0);
        gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p_1, gamma_0123_1);
        gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p_2, gamma_0123_2);
        gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p_3, gamma_0123_3);
        gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p_0, gamma_4567_0);
        gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p_1, gamma_4567_1);
        gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p_2, gamma_4567_2);
        gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p_3, gamma_4567_3);
    }
    _mm256_storeu_pd( &C[0 * rsC + 0 * csC], gamma_0123_0 );
    _mm256_storeu_pd( &C[0 * rsC + 1 * csC], gamma_0123_1 );
    _mm256_storeu_pd( &C[0 * rsC + 2 * csC], gamma_0123_2 );
    _mm256_storeu_pd( &C[0 * rsC + 3 * csC], gamma_0123_3 );
    _mm256_storeu_pd( &C[4 * rsC + 0 * csC], gamma_4567_0 );
    _mm256_storeu_pd( &C[4 * rsC + 1 * csC], gamma_4567_1 );
    _mm256_storeu_pd( &C[4 * rsC + 2 * csC], gamma_4567_2 );
    _mm256_storeu_pd( &C[4 * rsC + 3 * csC], gamma_4567_3 );
}

static void padMat(double* newMat, double* oldMat, int m, int n, int rsC, int csC) {
    int newM = ALIGN(m, _MR);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            *(newMat+i+newM*j) = *(oldMat+i*rsC+j*csC);
        }
    }
}

static void unpadMat(double* newMat, double* oldMat, int m, int n, int rsC, int csC) {
    int newM = ALIGN(m, _MR);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            *(oldMat+i*rsC+j*csC) = *(newMat+i+newM*j);
        }
    }
}