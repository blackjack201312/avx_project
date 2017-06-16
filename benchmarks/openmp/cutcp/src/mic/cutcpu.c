/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "atom.h"
#include "cutoff.h"

#undef DEBUG_PASS_RATE
#define CHECK_CYLINDER_CPU

#define CELLEN      4.f
#define INV_CELLEN  (1.f/CELLEN)

extern int cpu_compute_cutoff_potential_lattice(
		Lattice *lattice,                  /* the lattice */
		float cutoff,                      /* cutoff distance */
		Atoms *atoms                       /* array of atoms */
		)
{
	int latticeDimNx = lattice->dim.nx;
	int latticeDimNy = lattice->dim.ny;
	int latticeDimNz = lattice->dim.nz;
	float latticeDimLoX = lattice->dim.lo.x;
	float latticeDimLoY = lattice->dim.lo.y;
	float latticeDimLoZ = lattice->dim.lo.z;
	float latticeDimH = lattice->dim.h;
	int latLatticeSize = ((latticeDimNx * latticeDimNy * latticeDimNz) + 7) & ~ 7;
	float *latLattice;	// = lattice->lattice;

	float *latLatticePtr = lattice->lattice;
	latLattice = (float *) calloc(latLatticeSize, sizeof(float));
	Atom *atom = atoms->atoms;
	int natoms = atoms->size;
	float *atomsX = (float*) malloc(sizeof(float)*natoms); 
	float *atomsY = (float*) malloc(sizeof(float)*natoms); 
	float *atomsZ = (float*) malloc(sizeof(float)*natoms); 
	float *atomsQ = (float*) malloc(sizeof(float)*natoms); 
	int i;
	for(i = 0; i < natoms; i++)
	{
		atomsX[i] = atom[i].x;
		atomsY[i] = atom[i].y;
		atomsZ[i] = atom[i].z;
		atomsQ[i] = atom[i].q;
	}
	
	const char* env_iter = getenv("ITER");
	int iteration = (env_iter != NULL) ? atoi(env_iter) : 1;
	printf("[ITERATION NUM]:%d\n", iteration);
	#pragma offload target(mic:0)\
	in(latticeDimNx, latticeDimNy, latticeDimNz, latLatticeSize, cutoff, \
		latticeDimLoX, latticeDimLoY, latticeDimLoZ, latticeDimH, natoms, iteration) \
	out(latLattice:length(latLatticeSize) into(lattice->lattice))\
	in(atomsX:length(natoms)) \
	in(atomsY:length(natoms)) \
	in(atomsZ:length(natoms)) \
	in(atomsQ:length(natoms))
	{
		int iter;
		for (iter = 0; iter < iteration; iter++)
		{
			int nx = latticeDimNx;
			int ny = latticeDimNy;
			int nz = latticeDimNz;
			float xlo = latticeDimLoX;
			float ylo = latticeDimLoY;
			float zlo = latticeDimLoZ;
			float gridspacing = latticeDimH;
	
			const float a2 = cutoff * cutoff;
			const float inv_a2 = 1.f / a2;
			float s;
			const float inv_gridspacing = 1.f / gridspacing;
			const int radius = (int) ceilf(cutoff * inv_gridspacing) - 1;
			/* lattice point radius about each atom */
	
			int n;
			int i, j, k;
			int ia, ib, ic;
			int ja, jb, jc;
			int ka, kb, kc;
			int index;
			int koff, jkoff;
	
			float x, y, z, q;
			float dx, dy, dz;
			float dz2, dydz2, r2;
			float e;
			float xstart, ystart;
	
			float *pg;
	
			int gindex;
			int ncell, nxcell, nycell, nzcell;
			int *first, *next;
			float inv_cellen = INV_CELLEN;
			//Vec3 minext, maxext;		/* Extent of atom bounding box */
			float minextX, minextY, minextZ;
			float maxextX, maxextY, maxextZ;
			float xmin, ymin, zmin;
			float xmax, ymax, zmax;
	
			#if DEBUG_PASS_RATE
			unsigned long long pass_count = 0;
			unsigned long long fail_count = 0;
			#endif
	
			/* find min and max extent */
			//get_atom_extent(&minext, &maxext, atoms);
			int cnt;
			for (cnt = 0; cnt < natoms; cnt++)	
			{
				minextX = fminf(minextX, atomsX[cnt]);
				maxextX = fmaxf(maxextX, atomsX[cnt]);
				minextY = fminf(minextY, atomsY[cnt]);
				maxextY = fmaxf(maxextY, atomsY[cnt]);
				minextZ = fminf(minextZ, atomsZ[cnt]);
				maxextZ = fmaxf(maxextZ, atomsZ[cnt]);
			}


			/* number of cells in each dimension */
			nxcell = (int) floorf((maxextX - minextX) * inv_cellen) + 1;
			nycell = (int) floorf((maxextY - minextY) * inv_cellen) + 1;
			nzcell = (int) floorf((maxextZ - minextZ) * inv_cellen) + 1;
			ncell = nxcell * nycell * nzcell;
	
			/* allocate for cursor link list implementation */
			first = (int *) malloc(ncell * sizeof(int));
			for (gindex = 0;  gindex < ncell;  gindex++) 
			{
				first[gindex] = -1;
			}
			next = (int *) malloc(natoms * sizeof(int));
			for (n = 0;  n < natoms;  n++) 
			{
				next[n] = -1;
			}
	
			/* geometric hashing */
			for (n = 0;  n < natoms;  n++) 
			{
				if (0 == atomsQ[n]) continue;  /* skip any non-contributing atoms */
				i = (int) floorf((atomsX[n] - minextX) * inv_cellen);
				j = (int) floorf((atomsY[n] - minextY) * inv_cellen);
				k = (int) floorf((atomsZ[n] - minextZ) * inv_cellen);
				gindex = (k*nycell + j)*nxcell + i;
				next[n] = first[gindex];
				first[gindex] = n;
			}
	
			#pragma omp parallel for private (n, q, x, y, z, ic, jc, kc, ia, ib, ja, jb, ka, kb, \
				xstart, ystart, dz, k, koff, dz2, j, dy, jkoff,    \
				dydz2, dx, index, pg, i, r2, s, e)
			/* traverse the grid cells */
			for (gindex = 0;  gindex < ncell;  gindex++) 
			{
				for (n = first[gindex];  n != -1;  n = next[n]) 
				{
					x = atomsX[n] - xlo;
					y = atomsY[n] - ylo;
					z = atomsZ[n] - zlo;
					q = atomsQ[n];
	
					/* find closest grid point with position less than or equal to atom */
					ic = (int) (x * inv_gridspacing);
					jc = (int) (y * inv_gridspacing);
					kc = (int) (z * inv_gridspacing);
	
					/* find extent of surrounding box of grid points */
					ia = ic - radius;
					ib = ic + radius + 1;
					ja = jc - radius;
					jb = jc + radius + 1;
					ka = kc - radius;
					kb = kc + radius + 1;
	
					/* trim box edges so that they are within grid point lattice */
					if (ia < 0)   ia = 0;
					if (ib >= nx) ib = nx-1;
					if (ja < 0)   ja = 0;
					if (jb >= ny) jb = ny-1;
					if (ka < 0)   ka = 0;
					if (kb >= nz) kb = nz-1;
	
					/* loop over surrounding grid points */
					xstart = ia*gridspacing - x;
					ystart = ja*gridspacing - y;
					dz = ka*gridspacing - z;
					for (k = ka;  k <= kb;  k++, dz += gridspacing) 
					{
						koff = k*ny;
						dz2 = dz*dz;
						dy = ystart;
						for (j = ja;  j <= jb;  j++, dy += gridspacing) 
						{
							jkoff = (koff + j)*nx;
							dydz2 = dy*dy + dz2;
					#ifdef CHECK_CYLINDER_CPU
							if (dydz2 >= a2) continue;
					#endif
	
							dx = xstart;
							index = jkoff + ia;
							pg = latLattice + index; //lattice->lattice + index;
	
							//#if defined(__INTEL_COMPILER)  //this part is wrong
							//          for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
							//            r2 = dx*dx + dydz2;
							//            s = (1.f - r2 * inv_a2) * (1.f - r2 * inv_a2);
							//            e = q * (1/sqrtf(r2)) * s;
							//            *pg += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED!! */
							//          }
							//#else
							for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) 
							{
								r2 = dx*dx + dydz2;
								if (r2 >= a2)
								{
								#ifdef DEBUG_PASS_RATE
									fail_count++;
								#endif
									continue;
								}
								#ifdef DEBUG_PASS_RATE
								pass_count++;
								#endif
								s = (1.f - r2 * inv_a2);
								e = q * (1/sqrtf(r2)) * s * s;
	
								#pragma omp atomic
								*pg += e;
							}
							//#endif
						}
					} /* end loop over surrounding grid points */
	
				} /* end loop over atoms in a gridcell */
			} /* end loop over gridcells */
			/* free memory */
			free(next);
			free(first);
		}
	}

	/* For debugging: print the number of times that the test passed/failed */
#ifdef DEBUG_PASS_RATE
	printf ("Pass :%lld\n", pass_count);
	printf ("Fail :%lld\n", fail_count);
#endif
	return 0;
}
