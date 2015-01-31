
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


struct Interp
{

protected:

	const int nz, ny, nx;

	virtual void interp_dim(std::vector<double>& W,
							std::vector<int>& inds,
							const int n,
							const double z) = 0;

public:

	std::vector<double> Wz;
	std::vector<double> Wy;
	std::vector<double> Wx;
	std::vector<int>    zind;
	std::vector<int>    yind;
	std::vector<int>    xind;

	const int P;

	Interp(int nz_, int ny_, int nx_, int P_):
		nz(nz_), ny(ny_), nx(nx_),
		Wz(P_, 0), Wy(P_, 0), Wx(P_, 0),
		zind(P_, 0), yind(P_, 0), xind(P_, 0),
		P(P_)
	{
	}

	void interp(double zis, double yis, double xis)
	{

		interp_dim(Wz, zind, nz, zis);
		interp_dim(Wy, yind, ny, yis);
		interp_dim(Wx, xind, nx, xis);

	}

};


struct InterpB3: public Interp
{

private:

	int l, c, r; // Left, Center, Right
	double Delta, Delta2;

	void interp_dim(std::vector<double>& W,
					std::vector<int>& inds,
					const int n,
					const double z)
	{

		c = round(z);
		c = c==n ? 0:c;

		l = c==0 ? n-1:c-1;
		r = c==n-1 ? 0:c+1;

		inds[0]=l; inds[1]=c; inds[2]=r;

		Delta = z-round(z);
		Delta2 = Delta*Delta;
		W[0] = (1.-4*Delta+4*Delta2)/8.;
		W[1] = (3.-4*Delta2)/4.;
		W[2] = (1.+4*Delta+4*Delta2)/8.;

	}

public:

	InterpB3(int nz_, int ny_, int nx_):
		Interp(nz_, ny_, nx_, 3)
	{
	}

};

int roll(int z, int n)
{

	if(z<0)
		return n+z;
	else
		return z%n;

}

struct InterpB4: public Interp
{

private:

	int ll, l, r, rr;
	double x;

	void interp_dim(std::vector<double>& W,
					std::vector<int>& inds,
					const int n,
					const double z)
	{

		l  = roll(floor(z), n);
		ll = roll(l-1, n);
		r  = roll(l+1, n);
		rr = roll(r+1, n);

		inds[0]=ll; inds[1]=l;
		inds[2]=r;  inds[3]=rr;

		x = z-(floor(z)+.5);
		W[0] = (1+x*(-6+x*(12-8*x)))/48.;
		W[1] = (23+x*(-30+x*(-12+24*x)))/48.;
		W[2] = (23+x*(30+x*(-12-24*x)))/48.;
		W[3] = (1+x*(6+x*(12+8*x)))/48.;

	}

public:

	InterpB4(int nz_, int ny_, int nx_):
		Interp(nz_, ny_, nx_, 4)
	{
	}

};


struct InterpB5: public Interp
{

private:

	int ll, l, c, r, rr;
	double x, x2;

	void interp_dim(std::vector<double>& W,
					std::vector<int>& inds,
					const int n,
					const double z)
	{

		c  = roll(round(z), n); 
		l  = roll(c-1, n);
		ll = roll(l-1, n);
		r  = roll(c+1, n);
		rr = roll(r+1, n);

		inds[0]=ll; inds[1]=l;
		inds[2]=c;
		inds[3]=r;  inds[4]=rr;

		x  = z-round(z);
		x2 = x*x;
		W[0] = (1+x*(-8+x*(24+x*(-32+x*16))))/384;
		W[1] = (19+x*(-44+x*(24+x*(16-x*16))))/96;
		W[2] = (115+x2*(-120+48*x2))/192;
		W[3] = (19+x*(44+x*(24+x*(-16-x*16))))/96;
		W[4] = (1+x*(8+x*(24+x*(32+x*16))))/384;

	}

public:

	InterpB5(int nz_, int ny_, int nx_):
		Interp(nz_, ny_, nx_, 5)
	{
	}

};


template<typename T>
void interp_par(const int nz, const int ny, const int nx, const double *vals,
				const int N, const double *z, const double dz,
				const double *y, const double dy,
				const double *x, const double dx, double *c)
{

#pragma omp parallel
	{

		double zis, yis, xis;
		double tmp;
		const int zoff = nx*ny;

		T interp(nz, ny, nx);
		const int P = interp.P;

#pragma omp for
		for(int i=0; i<N; i++)
		{

			zis = z[i]/dz;
			yis = y[i]/dy;
			xis = x[i]/dx;

			interp.interp(zis, yis, xis);

			c[i] = 0;
			for(int iz=0; iz<P; iz++)
				for(int iy=0; iy<P; iy++)
				{
					tmp = interp.Wz[iz]*interp.Wy[iy];
					for(int ix=0; ix<P; ix++)
						c[i] += interp.Wx[ix]*tmp*vals[interp.zind[iz]*zoff
													   +interp.yind[iy]*nx
													   +interp.xind[ix]];
				}

		}
	}
}


void interp_par(const int nz, const int ny, const int nx, const double *vals,
				const int N, const double *z, const double dz,
				const double *y, const double dy,
				const double *x, const double dx, double *c, const int P)
{
	if(P==2)
		interp_cic_par(nz, ny, nx, vals, N, z, dz,
					   y, dy, x, dx, c);
	else if(P==3)
		interp_par<InterpB3>(nz, ny, nx, vals, N, z, dz,
							 y, dy, x, dx, c);
	else if(P==4)
		interp_par<InterpB4>(nz, ny, nx, vals, N, z, dz,
							 y, dy, x, dx, c);
	else if(P==5)
		interp_par<InterpB5>(nz, ny, nx, vals, N, z, dz,
							 y, dy, x, dx, c);

}


template<typename T>
void weight_par(const int nz, const int ny, const int nx, double *grid,
				const int N, const double *z, const double dz, const double *y, const double dy,
				const double *x, const double dx, const double *q)
{
	
#pragma omp parallel
	{

		double zis, yis, xis;
		double tmp, qi;
		const int zoff = nx*ny;

		T interp(nz, ny, nx);
		const int P = interp.P;

#pragma omp for
		for(int i=0; i<N; i++)
		{

			zis = z[i]/dz;
			yis = y[i]/dy;
			xis = x[i]/dx;

			interp.interp(zis, yis, xis);

			qi = q[i];
			for(int iz=0; iz<P; iz++)
				for(int iy=0; iy<P; iy++)
				{
					tmp = qi*interp.Wz[iz]*interp.Wy[iy];
					for(int ix=0; ix<P; ix++)
#pragma omp atomic
						grid[interp.zind[iz]*zoff
							 +interp.yind[iy]*nx
							 +interp.xind[ix]] += interp.Wx[ix]*tmp;
				}

		}
	}

}		


void weight_par(const int nz, const int ny, const int nx, double *grid,
				const int N, const double *z, const double dz, const double *y, const double dy,
				const double *x, const double dx, const double *q, const int P)
{

	if(P==2)
		weight_cic_par(nz, ny, nx, grid, N, z, dz, y, dy,
					   x, dx, q);
	else if(P==3)
		weight_par<InterpB3>(nz, ny, nx, grid, N, z, dz, y, dy,
							 x, dx, q);
	else if(P==4)
		weight_par<InterpB4>(nz, ny, nx, grid, N, z, dz, y, dy,
							 x, dx, q);
	else if(P==5)
		weight_par<InterpB5>(nz, ny, nx, grid, N, z, dz, y, dy,
							 x, dx, q);
	// Add exceptions here		

}
