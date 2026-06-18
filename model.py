"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: implement elementwise c[i] = a[i] + b[i]
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];

    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] *= scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            a[i] = expf(a[i]);
        }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    // TODO: compute the max of each row and write it to out[r].
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float max_val = matrix[r * cols];

        for (int c = 1; c < cols; c++) {
            float val = matrix[r * cols + c];
            if (val > max_val) {
                max_val = val;
            }
        }
        out[r] = max_val;
    }
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    // TODO: write out[r] = sum of matrix row r
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float sum_val = matrix[r * cols];

        for (int c = 1; c < cols; c++) {
            sum_val += matrix[r * cols + c];
        }

        out[r] = sum_val;
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    // TODO: return the dot product of a and b
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    // TODO: compute C = A * B for row-major matrices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        
        for (int t = 0; t < k; t++) {
            sum += a[row * k + t] * b[t * n + col];
        }
        c[row * n + col] = sum;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // TODO: write out[c*rows + r] = in[r*cols + c]
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < seq_len && j < seq_len) {
        float sum = 0.0f;

        for (int d = 0; d < head_dim; d++) {
            sum += q[i * head_dim + d] *
                   k[j * head_dim + d];
        }

        scores[i * seq_len + j] = 
            sum / sqrtf((float)head_dim);
    }
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float max_val = -INFINITY;

        // Find Max
        for (int c = 0; c < cols; c++) {
            max_val = fmaxf(max_val, matrix[row * cols + c]);
        }

        // Exponentiate and accumulate sum
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float v = expf(matrix[row * cols + c] - max_val);
            matrix[row * cols + c] = v;
            sum += v;
        }

        // Normalize
        for (int c = 0; c < cols; c++) {
            matrix[row * cols + c] /= sum;
        }
    }
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    // TODO: compute out[i, d] = sum_j p[i, j] * v[j, d]
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;

        for (int j = 0; j < seq_len; j++) {
            sum += p[i * seq_len + j] * 
                   v[j * head_dim + d];
        }
        out[i * head_dim + d] = sum;
    }
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    
    // Allocate score matrix
    float* d_scores;
    cudaMalloc(&d_scores,
                seq_len * seq_len * sizeof(float));
    
    // 1. QK^T / sqrt(d)
    dim3 score_threads(16, 16);
    dim3 score_blocks(
        (seq_len + score_threads.x - 1) / score_threads.x,
        (seq_len + score_threads.y - 1) / score_threads.y
    );

    qk_scores<<<score_blocks, score_threads>>>(
        d_q, d_k, d_scores,
        seq_len, head_dim
    );

    // 2. Softmax
    int softmax_threads = 256;

    softmax_rows<<<seq_len,
                softmax_threads,
                softmax_threads * sizeof(float)>>>(
        d_scores,
        seq_len,
        seq_len
    );

    // 3. P * V
    dim3 pv_threads(16, 16);
    dim3 pv_blocks(
        (head_dim + pv_threads.x - 1) / pv_threads.x,
        (seq_len + pv_threads.y - 1) / pv_threads.y
    );

    pv_matmul<<<pv_blocks, pv_threads>>>(
        d_scores,
        d_v,
        d_out,
        seq_len,
        head_dim
    );
    cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    return (new_val > old_max) ? new_val : old_max;
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return old_sum * correction + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int i = 0; i < head_dim; i++) {
        out_row[i] *= correction;
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    int tile_size = tile_rows * tile_cols;

    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int r = idx / tile_cols;
        int c = idx % tile_cols;

        int src_r = src_row_start + r;
        int src_c = src_col_start + c;

        if (src_r < src_rows && src_c < src_cols) {
            shared_dst[idx] = src[src_r * src_cols + src_c];
        } else {
            shared_dst[idx] = 0.0f;
        }
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    int num_scores = tile_q * tile_k;

    for (int idx = thread_id; idx < num_scores; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;

        float dot = 0.0f;
        
        for (int d = 0; d < head_dim; d++) {
            dot += q_tile[i * head_dim + d] *
                   k_tile[j * head_dim + d];
        }

        s_tile[i * tile_k + j] = scale * dot;
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, 
int tile_k, int thread_id, int num_threads) {
    for (int row = thread_id; row < tile_q; row += num_threads) {
        float mx = -INFINITY;

        for (int col = 0; col < tile_k; col++) {
            mx = fmaxf(mx, s_tile[row * tile_k + col]);
        }
        
        row_max_out[row] = mx;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    int tile_size = tile_q * tile_k;

    for (int idx = thread_id; idx < tile_size; idx += num_threads) {
        int r = idx / tile_k;
        int c = idx % tile_k;

        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int row = thread_id; row < tile_q; row += num_threads) {
        float sum = 0.0f;

        for (int col = 0; col < tile_k; col++) {
            sum += p_tile[row * tile_k + col];
        }

        row_sum_out[row] = sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    int num_outputs = tile_q * head_dim;

    for (int idx = thread_id; idx < num_outputs; idx += num_threads) {
        int r = idx / head_dim;
        int d = idx % head_dim;

        float sum = 0.0f;

        for (int k = 0; k < tile_k; k++) {
            sum += p_tile[r * tile_k + k] *
                   v_tile[k * head_dim + d];
        }

        out_acc[idx] += sum;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    int q_start = blockIdx.x * tile_q;
    if (q_start >= seq_len) return;

    extern __shared__ float smem[];

    float* q_tile = smem;
    float* k_tile = q_tile + tile_q * head_dim;
    float* v_tile = k_tile + tile_k * head_dim;
    float* s_tile = v_tile + tile_k * head_dim;

    float* out_acc = s_tile + tile_q * tile_k;
    
    float* run_max  = out_acc + tile_q * head_dim;
    float* run_sum  = run_max + tile_q;
    float* row_max  = run_sum + tile_q;
    float* row_sum  = row_max + tile_q;
 
  

    load_tile(q, q_tile,
              q_start, 0,
              seq_len, head_dim,
              tile_q, head_dim,
              tid, nthreads);
    
    for (int r = tid; r < tile_q; r += nthreads) {
        run_max[r] = -1e30f;
        run_sum[r] = 0.0f;
    }

    for (int i = tid; i < tile_q * head_dim; i += nthreads) {
        out_acc[i] = 0.0f;
    }

    __syncthreads();

    for (int k_start = 0; k_start < seq_len; k_start += tile_k) {

        load_tile(k, k_tile,
                  k_start, 0,
                  seq_len, head_dim,
                  tile_k, head_dim,
                  tid, nthreads);

        load_tile(v, v_tile,
                  k_start, 0,
                  seq_len, head_dim,
                  tile_k, head_dim,
                  tid, nthreads);
        
        __syncthreads();

        tile_scores(q_tile, k_tile, s_tile,
                    tile_q, tile_k, head_dim,
                    scale, tid, nthreads);

        __syncthreads();

        for (int idx = tid; idx < tile_q * tile_k; idx += nthreads) {
            int c = idx % tile_k;

            if (k_start + c >= seq_len) {
                s_tile[idx] = -1e30f;
            }
        }
        
        __syncthreads();

        tile_rowmax(s_tile, row_max,
                    tile_q, tile_k,
                    tid, nthreads);

        __syncthreads();


        for (int r = tid; r < tile_q; r += nthreads) {
            float old_max = run_max[r];
            float block_m = row_max[r];

            float new_m = online_max(old_max, block_m);
            float corr = correction_factor(old_max, new_m);

            rescale_output(out_acc + r * head_dim,
                           head_dim, corr);

            run_sum[r] = 
                update_running_sum(run_sum[r],
                                   corr,
                                   0.0f);
            
            run_max[r] = new_m;
        }
        
        __syncthreads();

        for (int idx = tid; idx < tile_q * tile_k; idx += nthreads) {
            int r = idx / tile_k;

            s_tile[idx] = expf(s_tile[idx] - run_max[r]);
        }

        __syncthreads();

        tile_rowsum(s_tile, row_sum,
                     tile_q, tile_k,
                     tid, nthreads);
        
        __syncthreads();

        for (int r = tid; r < tile_q; r += nthreads) {
            run_sum[r] += row_sum[r];
        }

        __syncthreads();

        accumulate_pv(s_tile,
                      v_tile,
                      out_acc,
                      tile_q,
                      tile_k,
                      head_dim,
                      tid,
                      nthreads);
    }

    for (int idx = tid; idx < tile_q * head_dim; idx += nthreads) {

        int r = idx / head_dim;
        int d = idx % head_dim;

        int global_row = q_start + r;

        if (global_row < seq_len) {
            out[global_row * head_dim + d] =
                out_acc[idx] / run_sum[r];
        }
    }
}

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

