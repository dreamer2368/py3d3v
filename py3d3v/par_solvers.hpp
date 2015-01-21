
#pragma once
#define _USE_MATH_DEFINES
#include <omp.h>
#include <math.h>
#include <stdlib.h>

void calc_E_short_range_par_gaussian(int N,
									 double* Ezp, const double* zp, double Lz,
									 double* Eyp, const double* yp, double Ly,
									 double* Exp, const double* xp, double Lx,
									 const double* q,
									 long N_cells, const long* cell_span, 
									 double rmax, double beta);

void calc_E_short_range_par_s2(int N,
							   double* Ezp, const double* zp, double Lz,
							   double* Eyp, const double* yp, double Ly,
							   double* Exp, const double* xp, double Lx,
							   const double* q,
							   long N_cells, const long* cell_span, 
							   double rmax, double beta);

void build_k2_lr_gaussian_optim_par(double* k2_vals,
									double* kz, int nkz, double dz,		
									double* ky, int nky, double dy,
									double* kx, int nkx, double dx,
									double beta);


struct Gaussian
{

private:

	// Beta and Beta^2
	double beta, beta2;
	// Constants in E
	double c, d, r2;

public:

	Gaussian(double beta_);

	double E(double r);

};

/*! S2 from Hockney and Eastwood
 */
struct S2
{

private:

	// Beta and Beta^2
	double beta;
	// Constants in E
	double c, d, fpii, b2, r2;

public:

	S2(double beta_);

	double E(double r);

};
