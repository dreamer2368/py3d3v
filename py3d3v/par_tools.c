
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
