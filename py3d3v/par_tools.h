
#pragma once

#include <omp.h>

void move_par(const int N, const double dt,
			  double *zp, const double *vz,
			  double *yp, const double *vy,
			  double *xp, const double *vx);

void accel_par(const int N, const double dt, const double *qm,
			   const double *Ez, double *vz,
			   const double *Ey, double *vy,
			   const double *Ex, double *vx);

