#include "assignment3.h"
#include "immintrin.h"
//all sizes in units of doubles
#define R_SIZE 4 //256 bits, 4 total doubles mr and nr
#define MAIN_MEMORY_SIZE (1UL << 32)
#define L3_CACHE_SIZE (1U << 21)
#define L2_CACHE_SIZE (1U << 15) // private L2 Caches, 8 cores
#define L1_CACHE_SIZE (1U << 12)
#define _MR 8
#define _NR 6
#define _NC 1026
#define _KC 256
#define _MC 96
#define ALIGN(size, alignment) (((size) + (alignment - 1)) / alignment * alignment)
#define THREAD_COUNT 8

double Bpacked[_KC * _NC] __attribute__((aligned(4096)));


void shpc_dgemm( int m, int n, int k,
                    double *A, int rsA, int csA,                                
                    double *B, int rsB, int csB,                                
                    double *C, int rsC, int csC )
{
    for (int j = 0; j < n; j+=_NC) {
        double* Cj = C+(csC * j);
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
            #pragma omp parallel for num_threads(THREAD_COUNT)
            for (int i = 0; i < m; i+=_MC) {
                //pack loops:
                double Apacked[_MC * _KC] __attribute__((aligned(4096)));
                double* padC = calloc(_NR *_MR, sizeof(double));
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
                        //fill in the contents of C
                        for (int j_pack = 0; j_pack < _NR && j + _j + j_pack < n; j_pack++) {
                            for (int i_pack = 0; i_pack < _MR && i + _i + i_pack < m; i_pack++) {
                                *(padC + i_pack + j_pack * _MR) = *(Cj+(i+_i+i_pack)*rsC+(_j+j_pack)*csC);
                            }
                        }
                        //do the uey
                        ukernel(&Apacked[_i*_KC], 1, _MR, &Bpacked[_j*_KC], _NR, 1, padC, 1, _MR);
                        //write back the good stuff
                        for (int j_pack = 0; j_pack < _NR && j + _j + j_pack < n; j_pack++) {
                            for (int i_pack = 0; i_pack < _MR && i + _i + i_pack < m; i_pack++) {
                                *(Cj+(i+_i+i_pack)*rsC+(_j+j_pack)*csC) = *(padC + i_pack + j_pack * _MR);
                            }
                        }
                    }
                }
                free(padC);
            }
        }
    }
}

void ukernel (double *A, int rsA, int csA, double* B, int rsB, int csB, double* C, int rsC, int csC) {
    __m256d gamma_0123_0 = _mm256_loadu_pd(&C[0 * rsC + 0 * csC]);
    __m256d gamma_0123_1 = _mm256_loadu_pd(&C[0 * rsC + 1 * csC]);
    __m256d gamma_0123_2 = _mm256_loadu_pd(&C[0 * rsC + 2 * csC]);
    __m256d gamma_0123_3 = _mm256_loadu_pd(&C[0 * rsC + 3 * csC]);
    __m256d gamma_0123_4 = _mm256_loadu_pd(&C[0 * rsC + 4 * csC]);
    __m256d gamma_0123_5 = _mm256_loadu_pd(&C[0 * rsC + 5 * csC]);

    __m256d gamma_4567_0 = _mm256_loadu_pd(&C[4 * rsC + 0 * csC]);
    __m256d gamma_4567_1 = _mm256_loadu_pd(&C[4 * rsC + 1 * csC]);
    __m256d gamma_4567_2 = _mm256_loadu_pd(&C[4 * rsC + 2 * csC]);
    __m256d gamma_4567_3 = _mm256_loadu_pd(&C[4 * rsC + 3 * csC]);
    __m256d gamma_4567_4 = _mm256_loadu_pd(&C[4 * rsC + 4 * csC]);
    __m256d gamma_4567_5 = _mm256_loadu_pd(&C[4 * rsC + 5 * csC]);

    // Matrix multiplication loop
    __m256d beta_p_n1;
    __m256d beta_p_n2;
    beta_p_n1 = _mm256_broadcast_sd(&B[0 * rsB + 0 * csB]); // n = 0

#pragma unroll 8
    for (int p = 0; p < _KC; p++) {
        __m256d alpha_0123_p = _mm256_loadu_pd(&A[0 * rsA + p * csA]);
        __m256d alpha_4567_p = _mm256_loadu_pd(&A[4 * rsA + p * csA]);

        beta_p_n2 = _mm256_broadcast_sd(&B[p * rsB + 1 * csB]); // n = 0
        gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p_n1, gamma_0123_0);
        gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p_n1, gamma_4567_0);

        beta_p_n1 = _mm256_broadcast_sd(&B[p * rsB + 2 * csB]); //n = 1
        gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p_n2, gamma_0123_1);
        gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p_n2, gamma_4567_1);

        beta_p_n2 = _mm256_broadcast_sd(&B[p * rsB + 3 * csB]); //n = 2
        gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p_n1, gamma_0123_2);
        gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p_n1, gamma_4567_2);


        beta_p_n1 = _mm256_broadcast_sd(&B[p * rsB + 4 * csB]); //n = 3
        gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p_n2, gamma_0123_3);
        gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p_n2, gamma_4567_3);

        beta_p_n2 = _mm256_broadcast_sd(&B[p * rsB + 5 * csB]); //n = 4
        gamma_0123_4 = _mm256_fmadd_pd(alpha_0123_p, beta_p_n1, gamma_0123_4);
        gamma_4567_4 = _mm256_fmadd_pd(alpha_4567_p, beta_p_n1, gamma_4567_4);

        beta_p_n1 = _mm256_broadcast_sd(&B[(p+1) * rsB + 0 * csB]); //n = 5
        gamma_0123_5 = _mm256_fmadd_pd(alpha_0123_p, beta_p_n2, gamma_0123_5);
        gamma_4567_5 = _mm256_fmadd_pd(alpha_4567_p, beta_p_n2, gamma_4567_5);
    }

    _mm256_storeu_pd( &C[0 * rsC + 0 * csC], gamma_0123_0 );
    _mm256_storeu_pd( &C[0 * rsC + 1 * csC], gamma_0123_1 );
    _mm256_storeu_pd( &C[0 * rsC + 2 * csC], gamma_0123_2 );
    _mm256_storeu_pd( &C[0 * rsC + 3 * csC], gamma_0123_3 );
    _mm256_storeu_pd( &C[0 * rsC + 4 * csC], gamma_0123_4 );
    _mm256_storeu_pd( &C[0 * rsC + 5 * csC], gamma_0123_5 );
    _mm256_storeu_pd( &C[4 * rsC + 0 * csC], gamma_4567_0 );
    _mm256_storeu_pd( &C[4 * rsC + 1 * csC], gamma_4567_1 );
    _mm256_storeu_pd( &C[4 * rsC + 2 * csC], gamma_4567_2 );
    _mm256_storeu_pd( &C[4 * rsC + 3 * csC], gamma_4567_3 );
    _mm256_storeu_pd( &C[4 * rsC + 4 * csC], gamma_4567_4 );
    _mm256_storeu_pd( &C[4 * rsC + 5 * csC], gamma_4567_5 );

}