#include "assignment3.h"

int kc_param = 5;
int nc_param = 5;
int mc_param = 5;

int main(int argc, char *argv[])
{
	int first, last, inc, nrepeats;
    
    int err = get_args( argc, argv, &nrepeats, &first, &last, &inc );
    if ( err != 0 ) return 1;

    test_gemm(nrepeats, first, last, inc);

    //printf("best params: kc: %d, mc: %d, nc: %d\n", best_kc, best_mc, best_nc);

    return 0;

}

