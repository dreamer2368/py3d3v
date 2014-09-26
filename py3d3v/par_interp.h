
#pragma once

#include <math.h>
#include <omp.h>

void interp_cic_par(const int nz, const int ny, const int nx, const double *vals,
					const int N, const double *z, const double *y, const double *x, double *c);


