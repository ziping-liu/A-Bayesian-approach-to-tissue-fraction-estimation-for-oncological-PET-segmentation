#include "gen_poisson_rv.h"

float gen_poisson_rv(float sinogram_data){
	
	float val;
	
	const gsl_rng_type *T;
	gsl_rng * r;

	gsl_rng_env_setup();


	struct timeval tv; // Seed generation based on time
	gettimeofday(&tv,0);
	unsigned long mySeed = tv.tv_sec + tv.tv_usec;

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r, mySeed);
	
	unsigned int k = gsl_ran_poisson(r, sinogram_data);
	
	val = (float)k;
	
	return val;
	
}
