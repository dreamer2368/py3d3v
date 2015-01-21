
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

void build_inf_lr_gaussian_optim_par(double* k2_vals,
									 double* kz, int nkz, double dz,		
									 double* ky, int nky, double dy,
									 double* kx, int nkx, double dx,
									 double beta);

void build_inf_lr_s2_optim_par(double* k2_vals,
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
	double c, d, e, r2;

public:

	Gaussian(double beta_);

	// Field from screen
	double E(double r);

	// Fourier transform at k2=k*k
	double g(double k2);

};

/*! S2 from Hockney and Eastwood
 */
struct S2
{

private:

	// Beta and Beta^2
	double beta;
	// Constants in E
	double c, d, e, fpii, b2, r2, kb2;

public:

	S2(double beta_);

	double E(double r);

	double g(double k2);

};
