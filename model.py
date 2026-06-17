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

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

