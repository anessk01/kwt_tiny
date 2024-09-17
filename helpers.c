// Author: Aness Al-Qawlaq 
// Date: 21/2/2024
// University College Dublin

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void computeMeanAndVariance(float* input, float* mean, float* variance, int dim, int input_size) {
    int num_samples = input_size / dim;

    // Calculate mean for each sample
    for (int sample = 0; sample < num_samples; ++sample) {
        mean[sample] = 0;
        for (int feature = 0; feature < dim; ++feature) {
            mean[sample] += input[sample * dim + feature];
        }
        mean[sample] /= dim;
    }

    // Calculate variance for each sample
    for (int sample = 0; sample < num_samples; ++sample) {
        variance[sample] = 0;
        for (int feature = 0; feature < dim; ++feature) {
            float diff = input[sample * dim + feature] - mean[sample];
            variance[sample] += diff * diff;
        }
        variance[sample] /= dim;
    }
}


void layerNorm(float* input, float* output, float* gamma, float* beta, int dim, int input_size) {
    // Calculate mean and variance for each sample in the batch
    float mean[input_size / dim];
    float variance[input_size / dim];

    computeMeanAndVariance(input, mean, variance, dim, input_size);

    // Normalize each element
    for (int i = 0; i < input_size; ++i) {
        int sample_index = i / dim; // Index of the sample in the batch
        float normalized_value = (input[i] - mean[sample_index]) / sqrt(variance[sample_index] + 1e-5);

        // Apply scale (gamma) and shift (beta)
        output[i] = gamma[i % dim] * normalized_value + beta[i % dim];
    }
}

// Helper function for matrix multiplication
void matrixMultiply(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    // printf("%d\n", A_rows * B_cols);
    for (int i = 0; i < A_rows * B_cols; i++) {
        C[i] = 0;
    }

    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            for (int k = 0; k < A_cols; k++) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

void chunk_qkv(float* result, float* qkv_new, int rows, int total_dim) {
    // rearrange qkv 
    int split_size = total_dim / 3;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < split_size; ++j) {
            qkv_new[i * split_size + j] = result[i * total_dim + j]; // q
            qkv_new[i * split_size + j + split_size*rows] = result[i * total_dim + split_size + j]; // k
            qkv_new[i * split_size + j + 2*split_size*rows] = result[i * total_dim + 2 * split_size + j]; // v
        }
    }
}


// Helper function to add bias with 1d flattened projection
void addBias(float* output, float* bias, int rows, int cols, int project_to) {
    int j=0;
    for(int a=0; a<project_to;a++){
        j = a%(rows*cols);
        output[a] += bias[j];
        // printf("added bias:%d=%f to output:%d\n", j, bias[j], a);
    }
}

// Linear layer
void linear(float* weights, float* bias, float* input, float* output, int A_rows, int A_cols, int B_cols, int bias_dim) {
    // Perform matrix multiplication input * weights
    matrixMultiply(input, weights, output, A_rows, A_cols, B_cols);

    // Add bias
    addBias(output, bias, 1, bias_dim, A_rows*B_cols);
}


// Helper function for GELU activation
void gelu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = 0.5 * input[i] * (1 + erf(input[i] / sqrt(2)));
    }
}


// helper to concatenate array
void concat_array(float* array1, float* array2, int array1_len, int array2_len, float* result){
    for(int i=0; i<array1_len; i++){
        result[i]=array1[i];
    }
    for(int j=0; j<array2_len; j++){
        result[j+array1_len]=array2[j];
    }
}

// helper to add and overwrite
void add_overwrite(float* source, float* to_add, int source_len){
    for(int i=0; i<source_len; i++){
        source[i] += to_add[i];
    }
}

// split into q,k,v
void splitIntoQKV(float* qkv, float* q, float* k, float* v, int batch_size, int seq_length, int heads, int dim_head) {
    int segment_length = batch_size * seq_length * heads * dim_head;

    for (int b = 0; b < batch_size; ++b) {
        for (int n = 0; n < seq_length; ++n) {
            for (int hd = 0; hd < heads; ++hd) {
                for (int d = 0; d < dim_head; ++d) {
                    int base_idx = b * seq_length * heads * dim_head + n * heads * dim_head + hd * dim_head;
                    int qkv_idx = base_idx + d;
                    int flat_idx = ((b * heads + hd) * seq_length + n) * dim_head + d;
                    // printf("q: %d = qkv: %d\n", flat_idx, qkv_idx);
                    // Assign values to q, k, v from qkv
                    q[flat_idx] = qkv[qkv_idx];
                    k[flat_idx] = qkv[qkv_idx + segment_length];
                    v[flat_idx] = qkv[qkv_idx + 2 * segment_length];
                }
            }
        }
    }
}

// non-accelerated version of softmax
void softmax(float* x, int length) {
    // unsigned int compute_cycles= 0;
    
    // find max value
    float max_val = x[0];
    for (int i = 1; i < length; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // reset_mcycle();

    float sum = 0.0;
    for (int i = 0; i < length; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < length; ++i) {
        x[i] /= sum;
    }
    // compute_cycles += get_mcycle(); 

    // x[0] = compute_cycles/90000;
}

// compute scaled dot product attention
void scaledDotProductAttention(float* q, float* k, float* v, float* attention_scores, float* output, int batch_size, int heads, int seq_length, int dim_head, float scale) {
    int depth = seq_length * dim_head;
    int index=0;
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < seq_length; ++i) {
                for (int j = 0; j < seq_length; ++j) {
                    float dot_product = 0.0;
                    for (int d = 0; d < dim_head; ++d) {
                        int index = b * seq_length * heads * dim_head + h * depth + i * dim_head + d;
                        int k_index = b * seq_length * heads * dim_head + h * depth + j * dim_head + d;
                        dot_product += q[index] * k[k_index];
                    }
                    attention_scores[b * heads * seq_length * seq_length + h * seq_length * seq_length + i * seq_length + j] = dot_product * scale;
                }
            }
            // Apply softmax to attention_scores
            for (int i = 0; i < seq_length; ++i) {
                softmax(&attention_scores[b * heads * seq_length * seq_length + h * seq_length * seq_length + i * seq_length], seq_length);
            }
            // Multiply by V
            for (int i = 0; i < seq_length; ++i) {
                for (int d = 0; d < dim_head; ++d) {
                    float weighted_sum = 0.0;
                    for (int j = 0; j < seq_length; ++j) {
                        int score_index = b * heads * seq_length * seq_length + h * seq_length * seq_length + i * seq_length + j;
                        int v_index = b * seq_length * heads * dim_head + h * depth + j * dim_head + d;
                        weighted_sum += attention_scores[score_index] * v[v_index];
                    }
                    index = b * (seq_length * heads * dim_head) + i * (heads * dim_head) + h * dim_head + d;
                    
                    output[index] = weighted_sum;
                }
            }
        }
    }
}

// final linear layer
void linear_final_attn(float* attention_output, float* out_weights, float* out_bias, float* linear_output, int batch_size, int seq_length, int inner_dim, int dim) {
    // for(int i=0;i<3072;i++){
    //     printf("%d: %f\n",i+1, attention_output[i]);
    // }
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            for (int j = 0; j < dim; j++) {
                float sum = 0.0;
                // printf("NEXT \n");
                for (int k = 0; k < inner_dim; k++) {
                    sum += attention_output[b * seq_length * inner_dim + s * inner_dim + k] * out_weights[k * dim + j];
                    // printf("multiplied %f by %f. Sum: %f\n", attention_output[b * seq_length * inner_dim + s * inner_dim + k], out_weights[k * dim + j], sum);
                }
                linear_output[b * seq_length * dim + s * dim + j] = sum + out_bias[j];
            }
        }
    }
}