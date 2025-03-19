#include "assignment3.h"

int test_gemm( int nrepeats, int first, int last, int inc)
{
	int size, irep;
	int m, n, k;
	int csA, csB, csC;
	int rsA, rsB, rsC;

	double *A, *B, *C, *Cref, *Cold;

	double done = 1.0;

	double t_ref = DBL_MAX;
	double t     = DBL_MAX;
	double t_start; 

	double gflops_ref, gflops;

	double diff, maxdiff = 0.0;

	printf( "%% --------- DGEMM --------- \n");
	printf( "data_dgemm" );
	printf( "( %4lu, 1:6 ) = [ %5lu %5lu %5lu %8.2f %8.2f %8.2f %15.4e ];\n",
						 ( unsigned long )( last - first)/inc + 1,
	        			 ( unsigned long )0,
	        			 ( unsigned long )0,
	        			 ( unsigned long )0, 0.0, 0.0, 0.0, 0.0 );
	for ( size=last; size>= first; size-=inc )
	{
    	/* we will only time cases where all three matrices are square */
    	m = n = k = size;
		csA = m; csB = k; csC = m;

		rsA = 1 ; rsB = 1, rsC = 1;

    	A = ( double * ) malloc( csA * k * sizeof( double ) );
    	B = ( double * ) malloc( csB * n * sizeof( double ) );
    	C = ( double * ) malloc( csC * n * sizeof( double ) );
    	Cold = ( double * ) malloc( csC * n * sizeof( double ) );
    	Cref = ( double * ) malloc( csC * n * sizeof( double ) );


		bli_drandm( 0, BLIS_DENSE, m, k, A, rsA, csA);
		bli_drandm( 0, BLIS_DENSE, k, n, B, rsB, csB);
		bli_drandm( 0, BLIS_DENSE, m, n, Cold, rsC, csC);


		for ( irep=0; irep<nrepeats; irep++ )
		{
			memcpy( Cref, Cold, csC * n * sizeof( double ) );

			t_start = bli_clock();
		
			bli_dgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,  
						m, n, k, &done, 
						A, rsA, csA, 
						B, rsB, csB, 
						&done, Cref, rsC, csC );	
			t_ref = bli_clock_min_diff( t_ref, t_start );
			
		}

		gflops_ref = 2.0 * m * n * k / ( t_ref * 1.0e9 );

		 
		for ( irep=0; irep<nrepeats; irep++ )
		{
			memcpy( C, Cold, csC * n * sizeof( double ) );

			t_start = bli_clock();
		
			shpc_dgemm( m, n, k, 
						A, rsA, csA, 
						B, rsB, csB, 
					    C, rsC, csC );	
			
			t = bli_clock_min_diff( t , t_start );
			
		}

		gflops = 2.0 * m * n * k / ( t * 1.0e9 );

		diff    = shpc_maxabsdiff( m, n, C, rsC, csC, Cref, rsC, csC );
        maxdiff = max ( diff, maxdiff );

		printf( "data_dgemm");
		printf( "( %4lu, 1:6 ) = [ %5lu %5lu %5lu %8.2f %8.2f %8.2f %15.4e ];\n",
		        ( unsigned long )(size - first)/inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )k,
		        ( unsigned long )n, gflops_ref, gflops, gflops/gflops_ref, diff );



		free( A );
		free( B );
		free( C );
		free( Cold );
		free( Cref );

	}

}

