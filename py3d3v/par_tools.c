
#include "par_tools.h"

void move_par(const int N, const double dt,
			  double *zp, const double *vz,
			  double *yp, const double *vy,
			  double *xp, const double *vx)
{

	int i;
	#pragma omp parallel for
	for(i=0; i<N; i++)
	{
		zp[i] = zp[i] + dt*vz[i];
		yp[i] = yp[i] + dt*vy[i];
		xp[i] = xp[i] + dt*vx[i];
	}

}


void accel_par(const int N, const double dt, const double *qm,
			   const double *Ez, double *vz,
			   const double *Ey, double *vy,
			   const double *Ex, double *vx)
{

	int i;
	#pragma omp parallel for
	for(i=0; i<N; i++)
	{
		double dtqmi = dt*qm[i];
		vz[i] = vz[i] + dtqmi*Ez[i];
		vy[i] = vy[i] + dtqmi*Ey[i];
		vx[i] = vx[i] + dtqmi*Ex[i];
	
	}
	
}


void scale_array(const int N, double *x, double sx)
{

	int i;
	#pragma omp parallel for
	for(i=0; i<N; i++)
		x[i] = x[i]*sx;

}


void scale_array_3_copy(const int N,
						const double *z, const double sz, double *zc,
						const double *y, const double sy, double *yc,
						const double *x, const double sx, double *xc)
{

	int i;
    #pragma omp parallel for
	for(i=0; i<N; i++)
	{
		zc[i] = z[i]*sz;
		yc[i] = y[i]*sy;
		xc[i] = x[i]*sx;
	}

}

void calc_E_short_range_par(int N,
							double* Ezp, const double* zp, double Lz,
							double* Eyp, const double* yp, double Ly,
							double* Exp, const double* xp, double Lx,
							const double* q, double rmax, double beta)
{

	double beta2 = beta*beta;
    double r2max = rmax*rmax;

	// Constants in short range force calculation
	double fourpii = 1./(4*M_PI);
    double c = 1./(2.*sqrt(M_PI*M_PI*M_PI));
	double d = sqrt(M_PI)/2.;

	int i, j;

	double dz, dy, dx, r2, r;
    double E, Ep, EpEr;

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
				r = sqrt(r2);
				E = c*(d*erf(r*beta)/r2-beta*exp(-beta2*r2)/r);
				Ep = fourpii/r2;
				EpEr = (Ep-E)/r;
				Ezp[i] +=  q[j]*dz*EpEr;
				Ezp[j] += -q[i]*dz*EpEr;
				Eyp[i] +=  q[j]*dy*EpEr;
				Eyp[j] += -q[i]*dy*EpEr;
				Exp[i] +=  q[j]*dx*EpEr;
				Exp[j] += -q[i]*dx*EpEr;
			}

		} // for j
	} // for i
}
