#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void
replicate0(int tot_size, char* flags_d) {
    int elems_per_block = (tot_size + gridDim.x - 1) / gridDim.x;
    int elems_per_thread = (elems_per_block + blockDim.x - 1) / blockDim.x;
    int base_index = blockIdx.x*elems_per_block + threadIdx.x*elems_per_thread;
    for (int i = base_index; i < base_index+elems_per_thread; i++) {
        if (i < tot_size) {
            flags_d[i] = 0;
        }
    }
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    int elems_per_block = (mat_rows+ gridDim.x -1) / gridDim.x;
    int elems_per_thread = (elems_per_block + blockDim.x) / blockDim.x;
    int base_index = blockIdx.x*elems_per_block + threadIdx.x*elems_per_thread;
    flags_d[0] = 1; // I assume that the matrix is never fully empty
    for (int i = base_index; i < base_index+elems_per_thread; i++) {
        if (i < mat_rows) {
            flags_d[mat_shp_sc_d[i]] = 1;
        }
    }
}

__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    int elems_per_block = (tot_size + gridDim.x - 1) / gridDim.x;
    int elems_per_thread = (elems_per_block + blockDim.x - 1) / blockDim.x;
    int base_index = blockIdx.x*elems_per_block + threadIdx.x*elems_per_thread;
    for (int i = base_index; i < base_index+elems_per_thread; i++) {
        if (i < tot_size) {
            tmp_pairs[i] = mat_vals[i]*vct[mat_inds[i]];
        }
    }
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    int elems_per_block = (mat_rows+ gridDim.x -1) / gridDim.x;
    int elems_per_thread = (elems_per_block + blockDim.x) / blockDim.x;
    int base_index = blockIdx.x*elems_per_block + threadIdx.x*elems_per_thread;
    for (int row = base_index; row < base_index+elems_per_thread; row++) {
        if (row < mat_rows) {
            res_vct_d[row] = tmp_scan[mat_shp_sc_d[row]-1];
        }
    }
}

#endif // SPMV_MUL_KERNELS
