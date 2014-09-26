
#include "par_interp.h"

void interp_cic_par(const int nz, const int ny, const int nx, const double *vals,
					const int N, const double *z, const double *y, const double *x, double *c)
{

    #pragma omp parallel
	{

	double xd, yd, zd;
	double c00, c01, c10, c11;
	double c0, c1;
	int x0, x1, y0, y1, z0, z1;
	double xd1m;
	const int zoff = nx*ny;

	int i;
    #pragma omp for private(i)
	for(i=0; i<N; i++)
	{
		x0 = (int)(floor(x[i]));
		x1 = (int)((x0+1)%nx);
		y0 = (int)(floor(y[i]));
		y1 = (int)((y0+1)%ny);
		z0 = (int)(floor(z[i]));
		z1 = (int)((z0+1)%nz);
		zd = (z[i]-z0);
		yd = (y[i]-y0);
		xd = (x[i]-x0);
		xd1m = 1.-xd;

		c00  = vals[z0*zoff+y0*nx+x0]*xd1m+vals[z0*zoff+y0*nx+x1]*xd;
		c10  = vals[z0*zoff+y1*nx+x0]*xd1m+vals[z0*zoff+y1*nx+x1]*xd;
		c01  = vals[z1*zoff+y0*nx+x0]*xd1m+vals[z1*zoff+y0*nx+x1]*xd;
		c11  = vals[z1*zoff+y1*nx+x0]*xd1m+vals[z1*zoff+y1*nx+x1]*xd;

		c0   = c00*(1.-yd) + c10*yd;
		c1   = c01*(1.-yd) + c11*yd;

		c[i] = c0*(1.-zd) + c1*zd;

	}
	}
}
