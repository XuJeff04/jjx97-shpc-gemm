#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "blis.h"
#include "src.h"
#include "test.h"
#include "util.h"


#define dabs( x ) ( (x) < 0 ? -(x) : x )
#define max(x, y) (((x) > (y)) ? (x) : (y))


extern int kc_param;
extern int nc_param;
extern int mc_param;