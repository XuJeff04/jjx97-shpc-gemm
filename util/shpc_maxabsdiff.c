#include "assignment3.h"

double shpc_maxabsdiff( int m, int n, double *A, int rsA, int csA,
                                      double *B, int rsB, int csB )
    /* 
       shpc_maxabsdif returns the maximum absolute difference over
       corresponding elements of matrices A and B.
     */
{
    double diff=0.0;
    double temp;
    int  i, j;

    for ( i=0; i<m; i++ )
    {
        for ( j=0; j<n; j++ )
        {
            temp = dabs( A[ i*rsA  + j*csA ] - B[ i*rsB + j*csB ] );
            if ( temp  > diff )
                diff = temp;
        }
    }
    return diff;
}
