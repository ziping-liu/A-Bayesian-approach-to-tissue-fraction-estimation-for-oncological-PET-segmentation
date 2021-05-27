#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Method to reconstruct by applying OSEM algorithm

float* osem_reconstruct(float * sinogram, float * H_matrix, int p_num, int phi_num, int N, int subset_num, int iter_num);