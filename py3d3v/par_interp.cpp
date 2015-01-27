
#include "par_interp.hpp"

void interp_cic_par(const int nz, const int ny, const int nx, const double *vals,
					const int N, const double *z, const double dz,
					const double *y, const double dy,
					const double *x, const double dx, double *c)
{

#pragma omp parallel
	{

		double xd, yd, zd;
		double zis, yis, xis;
		double c00, c01, c10, c11;
		double c0, c1;
		int x0, x1, y0, y1, z0, z1;
		double xd1m;
		const int zoff = nx*ny;

		int i;
#pragma omp for private(i)
		for(i=0; i<N; i++)
		{
			xis = x[i]/dx;
			yis = y[i]/dy;
			zis = z[i]/dz;
			x0 = (int)(floor(xis));
			x1 = (int)((x0+1)%nx);
			y0 = (int)(floor(yis));
			y1 = (int)((y0+1)%ny);
			z0 = (int)(floor(zis));
			z1 = (int)((z0+1)%nz);
			zd = (zis-z0);
			yd = (yis-y0);
			xd = (xis-x0);
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


void weight_cic_par(const int nz, const int ny, const int nx, double *grid,
					const int N, const double *z, const double dz, const double *y, const double dy,
					const double *x, const double dx, const double *q)
{

#pragma omp parallel
	{
		double xd, yd, zd, qi;
		double zis, yis, xis;
		int x0, x1, y0, y1, z0, z1;

		const int zoff = nx*ny;
		double tmp;
		int i;
#pragma omp for private(i)
		for(i=0; i<N; i++)
		{
			xis = x[i]/dx;
			yis = y[i]/dy;
			zis = z[i]/dz;
			z0 = (int)(floor(zis));
			z1 = (int)((z0+1)%nz);
			y0 = (int)(floor(yis));
			y1 = (int)((y0+1)%ny);
			x0 = (int)(floor(xis));
			x1 = (int)((x0+1)%nx);
			zd = (zis-z0);
			yd = (yis-y0);
			xd = (xis-x0);
			qi = q[i];
    
			tmp = qi*(1-xd)*(1-yd)*(1-zd);
#pragma omp atomic
			grid[z0*zoff+y0*nx+x0] += tmp;
			tmp = qi*xd*(1-yd)*(1-zd);
#pragma omp atomic
			grid[z0*zoff+y0*nx+x1] += tmp;
			tmp = qi*(1-xd)*yd*(1-zd);
#pragma omp atomic
			grid[z0*zoff+y1*nx+x0] += tmp;
			tmp = qi*xd*yd*(1-zd);
#pragma omp atomic
			grid[z0*zoff+y1*nx+x1] += tmp;
			tmp = qi*(1-xd)*(1-yd)*zd;
#pragma omp atomic
			grid[z1*zoff+y0*nx+x0] += tmp;
			tmp = qi*xd*(1-yd)*zd;
#pragma omp atomic
			grid[z1*zoff+y0*nx+x1] += tmp;
			tmp = qi*(1-xd)*yd*zd;
#pragma omp atomic
			grid[z1*zoff+y1*nx+x0] += tmp;
			tmp = qi*xd*yd*zd;
#pragma omp atomic
			grid[z1*zoff+y1*nx+x1] += tmp;
		}
	}
}		

void interp_b3_par(const int nz, const int ny, const int nx, const double *vals,
				   const int N, const double *z, const double dz,
				   const double *y, const double dy,
				   const double *x, const double dx, double *c)
{

#pragma omp parallel
	{

		int zc, yc, xc; // Center point
		int zl, yl, xl; // Left point
		int zr, yr, xr; // Right point
		std::vector<double> Wz(3, 0);
		std::vector<double> Wy(3, 0);
		std::vector<double> Wx(3, 0);
		std::vector<int>    zind(3, 0);
		std::vector<int>    yind(3, 0);
		std::vector<int>    xind(3, 0);
		double zis, yis, xis;
		double Delta, Delta2, tmp;
		const int zoff = nx*ny;

#pragma omp for
		for(int i=0; i<N; i++)
		{

			zis = z[i]/dz;
			yis = y[i]/dy;
			xis = x[i]/dx;

			zc = round(zis);
			zc = zc==nz ? 0:zc;
			yc = round(yis);
			yc = yc==ny ? 0:yc;
			xc = round(xis);
			xc = xc==nx ? 0:xc;

			zl = zc==0 ? nz-1:zc-1;
			yl = yc==0 ? ny-1:yc-1;
			xl = xc==0 ? nx-1:xc-1;

			zr = zc==nz-1 ? 0:zc+1;
			yr = yc==ny-1 ? 0:yc+1;
			xr = xc==nx-1 ? 0:xc+1;

			zind[0] = zl; zind[1] = zc; zind[2] = zr;
			yind[0] = yl; yind[1] = yc; yind[2] = yr;
			xind[0] = xl; xind[1] = xc; xind[2] = xr;

			Delta = zis-round(zis);
			Delta2 = Delta*Delta;
			Wz[0] = (1.-4*Delta+4*Delta2)/8.;
			Wz[1] = (3.-4*Delta2)/4.;
			Wz[2] = (1.+4*Delta+4*Delta2)/8.;

			Delta = yis-round(yis);
			Delta2 = Delta*Delta;
			Wy[0] = (1.-4*Delta+4*Delta2)/8.;
			Wy[1] = (3.-4*Delta2)/4.;
			Wy[2] = (1.+4*Delta+4*Delta2)/8.;

			Delta = xis-round(xis);
			Delta2 = Delta*Delta;
			Wx[0] = (1.-4*Delta+4*Delta2)/8.;
			Wx[1] = (3.-4*Delta2)/4.;
			Wx[2] = (1.+4*Delta+4*Delta2)/8.;
			
			c[i] = 0;
			for(int iz=0; iz<3; iz++)
				for(int iy=0; iy<3; iy++)
				{
					tmp = Wz[iz]*Wy[iy];
					for(int ix=0; ix<3; ix++)
						c[i] += Wx[ix]*tmp*vals[zind[iz]*zoff+yind[iy]*nx+xind[ix]];
				}

		}
	}
}

void weight_b3_par(const int nz, const int ny, const int nx, double *grid,
				   const int N, const double *z, const double dz, const double *y, const double dy,
				   const double *x, const double dx, const double *q)
{
	
#pragma omp parallel
	{

		int zc, yc, xc; // Center point
		int zl, yl, xl; // Left point
		int zr, yr, xr; // Right point
		std::vector<double> Wz(3, 0);
		std::vector<double> Wy(3, 0);
		std::vector<double> Wx(3, 0);
		std::vector<int>    zind(3, 0);
		std::vector<int>    yind(3, 0);
		std::vector<int>    xind(3, 0);
		double zis, yis, xis;
		double Delta, Delta2, tmp, qi;
		const int zoff = nx*ny;

#pragma omp for
		for(int i=0; i<N; i++)
		{

			zis = z[i]/dz;
			yis = y[i]/dy;
			xis = x[i]/dx;

			zc = round(zis);
			zc = zc==nz ? 0:zc;
			yc = round(yis);
			yc = yc==ny ? 0:yc;
			xc = round(xis);
			xc = xc==nx ? 0:xc;

			zl = zc==0 ? nz-1:zc-1;
			yl = yc==0 ? ny-1:yc-1;
			xl = xc==0 ? nx-1:xc-1;

			zr = zc==nz-1 ? 0:zc+1;
			yr = yc==ny-1 ? 0:yc+1;
			xr = xc==nx-1 ? 0:xc+1;

			zind[0] = zl; zind[1] = zc; zind[2] = zr;
			yind[0] = yl; yind[1] = yc; yind[2] = yr;
			xind[0] = xl; xind[1] = xc; xind[2] = xr;

			Delta = zis-round(zis);
			Delta2 = Delta*Delta;
			Wz[0] = (1.-4*Delta+4*Delta2)/8.;
			Wz[1] = (3.-4*Delta2)/4.;
			Wz[2] = (1.+4*Delta+4*Delta2)/8.;

			Delta = yis-round(yis);
			Delta2 = Delta*Delta;
			Wy[0] = (1.-4*Delta+4*Delta2)/8.;
			Wy[1] = (3.-4*Delta2)/4.;
			Wy[2] = (1.+4*Delta+4*Delta2)/8.;

			Delta = xis-round(xis);
			Delta2 = Delta*Delta;
			Wx[0] = (1.-4*Delta+4*Delta2)/8.;
			Wx[1] = (3.-4*Delta2)/4.;
			Wx[2] = (1.+4*Delta+4*Delta2)/8.;
			
			qi = q[i];
			for(int iz=0; iz<3; iz++)
				for(int iy=0; iy<3; iy++)
				{
					tmp = qi*Wz[iz]*Wy[iy];
					for(int ix=0; ix<3; ix++)
#pragma omp atomic
						grid[zind[iz]*zoff+yind[iy]*nx+xind[ix]] += Wx[ix]*tmp;
				}

		}
	}

}		
