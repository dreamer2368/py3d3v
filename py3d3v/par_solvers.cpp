
#include "par_solvers.hpp"

Gaussian::Gaussian(double beta_):
	beta(beta_)
{
	beta2 = beta*beta;
	c = 1./(2.*sqrt(M_PI*M_PI*M_PI));
	d = sqrt(M_PI)/2.;
	e = -1./(beta*beta)/4.;
}

double Gaussian::E(double r)
{
	r2 = r*r;
	return c*(d*erf(r*beta)/r2-beta*exp(-beta2*r2)/r);
}

double Gaussian::g(double k2)
{
	return exp(e*k2);
}

S2::S2(double beta_):
	beta(beta_)
{
	c = 4./(M_PI*beta*beta*beta*beta);
	d = 2*beta;
	e = 12./pow(beta/2., 4);
	b2 = beta/2.;
	fpii = 1./(4*M_PI);
}

double S2::E(double r)
{
	r2 = r*r;
	if(r<b2)
		return c*(d*r-3*r2);
	else
		return fpii*r2;
}

double S2::g(double k2)
{
	kb2 = sqrt(k2)*b2;
	return e*(2.-2.*cos(kb2)-kb2*sin(kb2))/(k2*k2);
}

template<typename T>
void calc_E_short_range_par_single(int N,
								   double* Ezp, const double* zp, double Lz,
								   double* Eyp, const double* yp, double Ly,
								   double* Exp, const double* xp, double Lx,
								   const double* q, double rmax, double beta)
{

    double r2max = rmax*rmax;
	double fourpii = 1./(4*M_PI);

	T G(beta);

	int i, j;
    #pragma omp parallel private(i,j)
	{
		double dz, dy, dx, r2, r;
		double E, Ep, EpEr, tmp;

        #pragma omp for schedule(dynamic)
		for(i=0; i<N-1; i++)
		{
			for(j=i+1; j<N; j++)
			{
				dz = zp[i]-zp[j];
				dy = yp[i]-yp[j];
				dx = xp[i]-xp[j];
				if(fabs(dz)>rmax)
				{
					if(     fabs(dz-Lz)<rmax) dz = dz-Lz;
					else if(fabs(dz+Lz)<rmax) dz = dz+Lz;
				}
				if(fabs(dy)>rmax)
				{
					if(     fabs(dy-Ly)<rmax) dy = dy-Ly;
					else if(fabs(dy+Ly)<rmax) dy = dy+Ly;
				}
				if(fabs(dx)>rmax)
				{
					if(     fabs(dx-Lx)<rmax) dx = dx-Lx;
					else if(fabs(dx+Lx)<rmax) dx = dx+Lx;
				}

				r2 = dz*dz+dy*dy+dx*dx;
				if(r2<r2max && r2>0.)
				{

					r    = sqrt(r2);
					E    = G.E(r);
					Ep   = fourpii/r2;
					EpEr = (Ep-E)/r;

					tmp = q[j]*dz*EpEr;
					#pragma omp atomic
					Ezp[i] += tmp;

					tmp = -q[i]*dz*EpEr;
					#pragma omp atomic
					Ezp[j] += tmp;

					tmp = q[j]*dy*EpEr;
					#pragma omp atomic
					Eyp[i] += tmp;

					tmp = -q[i]*dy*EpEr;
					#pragma omp atomic
					Eyp[j] += tmp;

					tmp = q[j]*dx*EpEr;
					#pragma omp atomic
					Exp[i] += tmp;

					tmp = -q[i]*dx*EpEr;
					#pragma omp atomic
					Exp[j] += tmp;

				}

			} // for j
		} // for i
	}
}


template<typename T>
void calc_E_short_range_par(int N,
							double* Ezp, const double* zp, double Lz,
							double* Eyp, const double* yp, double Ly,
							double* Exp, const double* xp, double Lx,
							const double* q, double rmax, double beta)
{

    double r2max = rmax*rmax;
	double fourpii = 1./(4*M_PI);

	T G(beta);

	double dz, dy, dx, r2, r;
	double E, Ep, EpEr;

	for(int i=0; i<N-1; i++)
	{
		for(int j=i+1; j<N; j++)
		{
			dz = zp[i]-zp[j];
			dy = yp[i]-yp[j];
			dx = xp[i]-xp[j];

			r2 = dz*dz+dy*dy+dx*dx;
			if(r2<r2max && r2>0.)
			{

				r    = sqrt(r2);
				E    = G.E(r);
				Ep   = fourpii/r2;
				EpEr = (Ep-E)/r;

				Ezp[i] += q[j]*dz*EpEr;
				Ezp[j] += -q[i]*dz*EpEr;
				Eyp[i] += q[j]*dy*EpEr;
				Eyp[j] += -q[i]*dy*EpEr;
				Exp[i] += q[j]*dx*EpEr;
				Exp[j] += -q[i]*dx*EpEr;

			}

		} // for j
	} // for i
}

template<typename T>
void calc_E_short_range_par(int N1, int N2,
							double* Ezp, const double* zp, double Lz, double sz,
							double* Eyp, const double* yp, double Ly, double sy,
							double* Exp, const double* xp, double Lx, double sx,
							const double* zp2, const double* yp2,
							const double* xp2, const double* q2,
							double rmax, double beta)
{

    double r2max = rmax*rmax;
	double fourpii = 1./(4*M_PI);

	T G(beta);

	double dz, dy, dx, r2, r;
	double E, Ep, EpEr;

	for(int i=0; i<N1; i++)
	{
		for(int j=0; j<N2; j++)
		{
			dz = zp[i]-(zp2[j]+sz);
			dy = yp[i]-(yp2[j]+sy);
			dx = xp[i]-(xp2[j]+sx);

			r2 = dz*dz+dy*dy+dx*dx;
			if(r2<r2max && r2>0.)
			{

				r    = sqrt(r2);
				E    = G.E(r);
				Ep   = fourpii/r2;
				EpEr = (Ep-E)/r;
					
				Ezp[i] += q2[j]*dz*EpEr;
				Eyp[i] += q2[j]*dy*EpEr;
				Exp[i] += q2[j]*dx*EpEr;

			}

		} // for j
	} // for i
}

template<typename T>
void calc_E_short_range_par_cells(int N,
								  double* Ezp, const double* zp, double Lz,
								  double* Eyp, const double* yp, double Ly,
								  double* Exp, const double* xp, double Lx,
								  const double* q,
								  long N_cells, const long* cell_span, 
								  double rmax, double beta)
{

	const int N_tot = N_cells*N_cells*N_cells;
	const int N_cells2 = N_cells*N_cells;

#pragma omp parallel
	{
		int i, j, k, cs, ce, Np;
		int r_loc, rcs, rce, Npr;
		int rk, rj, ri;
		double sz, sy, sx;

#pragma omp for schedule(dynamic)
		for(int loc=0; loc<N_tot; loc++)
		{

			k = (int)(floor(loc/(N_cells2)));
			j = (int)(floor((loc-k*N_cells2)/N_cells));
			i = (int)((loc-k*N_cells2-j*N_cells));
        
			cs = cell_span[loc];
			ce = cell_span[loc+1];
			Np = ce-cs;

			if(Np>0)
				calc_E_short_range_par<T>(Np,
										  &Ezp[cs], &zp[cs], Lz, 
										  &Eyp[cs], &yp[cs], Ly,
										  &Exp[cs], &xp[cs], Lx, 
										  &q[cs], rmax, beta);

			for(int rk0=-1; rk0<=1; rk0++)
				for(int rj0=-1; rj0<=1; rj0++)
					for(int ri0=-1; ri0<=1; ri0++)
					{
		
						if((k+rk0)<0)
						{
							rk = N_cells-1;
							sz = -Lz;
						}
						else if((k+rk0)>=N_cells)
						{
							rk = 0;
							sz = +Lz;
						}
						else
						{
							rk = k+rk0;
							sz = 0.;
						}

						if((j+rj0)<0)
						{
							rj = N_cells-1;
							sy = -Ly;
						}
						else if((j+rj0)>=N_cells)
						{
							rj = 0;
							sy = +Ly;
						}
						else
						{
							rj = j+rj0;
							sy = 0;
						}

						if((i+ri0)<0)
						{
							ri = N_cells-1;
							sx = -Lx;
						}
						else if((i+ri0)>=N_cells)
						{
							ri = 0;
							sx = +Lx;
						}
						else
						{
							ri = i+ri0;
							sx = 0.;
						}
						
						r_loc = ri+rj*N_cells+rk*N_cells2;

						rcs = cell_span[r_loc];
						rce = cell_span[r_loc+1];
						Npr = rce-rcs;

						if(rk0!=0 || rj0!=0 || ri0!=0)
							if(Np>0 && Npr>0)
								calc_E_short_range_par<T>(Np, Npr,
														  &Ezp[cs], &zp[cs], Lz, sz,
														  &Eyp[cs], &yp[cs], Ly, sy,
														  &Exp[cs], &xp[cs], Lx, sx,
														  &zp[rcs], &yp[rcs], &xp[rcs],
														  &q[rcs], rmax, beta);
					}

		}
	}
}

template<typename T>
void calc_E_short_range_par_base(int N,
								 double* Ezp, const double* zp, double Lz,
								 double* Eyp, const double* yp, double Ly,
								 double* Exp, const double* xp, double Lx,
								 const double* q,
								 long N_cells, const long* cell_span, 
								 double rmax, double beta)
{

	if(N_cells>1)
		calc_E_short_range_par_cells<T>(N, Ezp, zp, Lz, Eyp, yp, Ly,
										Exp, xp, Lx, q, N_cells, cell_span,
										rmax, beta);
	else
		calc_E_short_range_par_single<T>(N, Ezp, zp, Lz, Eyp, yp, Ly,
										 Exp, xp, Lx, q, rmax, beta);

}


void calc_E_short_range_par_gaussian(int N,
									 double* Ezp, const double* zp, double Lz,
									 double* Eyp, const double* yp, double Ly,
									 double* Exp, const double* xp, double Lx,
									 const double* q,
									 long N_cells, const long* cell_span, 
									 double rmax, double beta)
{

	calc_E_short_range_par_base<Gaussian>(N, Ezp, zp, Lz, Eyp, yp, Ly,
										  Exp, xp, Lx, q, N_cells, cell_span,
										  rmax, beta);

}

void calc_E_short_range_par_s2(int N,
							   double* Ezp, const double* zp, double Lz,
							   double* Eyp, const double* yp, double Ly,
							   double* Exp, const double* xp, double Lx,
							   const double* q,
							   long N_cells, const long* cell_span, 
							   double rmax, double beta)
{

	calc_E_short_range_par_base<S2>(N, Ezp, zp, Lz, Eyp, yp, Ly,
									Exp, xp, Lx, q, N_cells, cell_span,
									rmax, beta);

}

// equivalent to dif(k*d/2)
double difh(double ki, double d)
{

    if(ki!=0)
        return sin(ki*d/2.)/(ki*d/2.);
    else
        return 1.;
}

template<typename T>
void build_inf_lr_optim_par(double* inf_vals,
							double* kz, int nkz, double dz,		
							double* ky, int nky, double dy,
							double* kx, int nkx, double dx,
							double beta)
{

	T screen(beta);
    const double msz = 2*M_PI/dz;
    const double msy = 2*M_PI/dy;
    const double msx = 2*M_PI/dx;

#pragma omp parallel
	{

		double k2p;
		double ex;
		double kzi, kyi, kxi;
		double kzim, kyim, kxim;
		double kzim2, kyim2;

		double Uk, Usum;
		double Ukzim, Ukyim;
		double Dkz, Dky, Dkx;
		double Dkz2, Dky2;
		double num, numz, numy, numx; 
		double denom;
		double Rz, Ry, Rx;

		// The iz,iy,ix loops are over the values of k in fourier
		// space mesh
		// The loops over i,j,k are the k+2pim/h sums
#pragma omp for
		for(int iz=0; iz<nkz; iz++)
		{
			kzi  = kz[iz];
			Dkz = sin(kzi*dz)/dz;
			Dkz2 = Dkz*Dkz;
			for(int iy=0; iy<nky; iy++)
			{
				kyi  = ky[iy];
				Dky = sin(kyi*dy)/dy;
				Dky2 = Dky*Dky;
				for(int ix=0; ix<nkx; ix++)
				{
					kxi = kx[ix];
					// Sum over k+2pim/h
					if(iz!=0 || iy!=0 || ix!=0)
					{
                    
						Dkx = sin(kxi*dx)/dx;
						Usum = 0.;
						numz = numy = numx = 0.;
						for(int i=-2; i<3; i++)
						{
							kzim = kzi+i*msz;
							kzim2 = kzim*kzim;
							Ukzim = difh(kzim, dz);
							for(int j=-2; j<3; j++)
							{
								kyim = kyi+j*msy;
								kyim2 = kyim*kyim;
								Ukyim = difh(kyim, dy);
								for(int k=-2; k<3; k++)
								{
								
									kxim = kxi+k*msx;
									Uk = Ukzim*Ukyim*difh(kxim, dx);
									Uk = Uk*Uk*Uk*Uk;
									Usum += Uk;

									k2p = kzim2+kyim2+kxim*kxim;
									if(k2p!=0.)
									{
										ex = screen.g(k2p)/k2p;
										Rz = kzim*ex;
										Ry = kyim*ex;
										Rx = kxim*ex;

										numz += Rz*Uk;
										numy += Ry*Uk;
										numx += Rx*Uk;
									}
								}
							}
						}

						num = numz*Dkz+numy*Dky+numx*Dkx;
						denom = (Dkz2+Dky2+Dkx*Dkx)*Usum*Usum;

						inf_vals[iz*nky*nkx+iy*nkx+ix] = num/denom;
					} // Sum over m
				}
			}

		} // Sum over k
	}// omp parallel
}

void build_inf_lr_gaussian_optim_par(double* inf_vals,
									 double* kz, int nkz, double dz,		
									 double* ky, int nky, double dy,
									 double* kx, int nkx, double dx,
									 double beta)
{

	build_inf_lr_optim_par<Gaussian>(inf_vals, kz, nkz, dz,		
									 ky, nky, dy, kx, nkx, dx,
									 beta);
	
}

void build_inf_lr_s2_optim_par(double* inf_vals,
							   double* kz, int nkz, double dz,		
							   double* ky, int nky, double dy,
							   double* kx, int nkx, double dx,
							   double beta)
{

	build_inf_lr_optim_par<S2>(inf_vals, kz, nkz, dz,		
							   ky, nky, dy, kx, nkx, dx,
							   beta);
	
}
