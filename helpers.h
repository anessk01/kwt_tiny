// Author: Aness Al-Qawlaq 
// Date: 21/2/2024
// University College Dublin

#include <math.h>
#include <stdint.h>

void computeMeanAndVariance(float* input, float* mean, float* variance, int dim, int input_size);
void layerNorm(float* input, float* output, float* gamma, float* beta, int dim, int input_size);

void matrixMultiply(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols);
void chunk_qkv(float* result, float* qkv_new, int rows, int total_dim);
void addBias(float* output, float* bias, int rows, int cols, int project_to);
void linear(float* weights, float* bias, float* input, float* output, int A_rows, int A_cols, int B_cols, int bias_dim);
void gelu(float* input, int size);

void concat_array(float* array1, float* array2, int array1_len, int array2_len, float* result);
void add_overwrite(float* source, float* to_add, int source_len);

void splitIntoQKV(float* qkv_output, float* q, float* k, float* v, int batch_size, int seq_length, int heads, int dim_head);
void softmax(float* x, int length);

void scaledDotProductAttention(float* q, float* k, float* v, float* attention_scores, float* output, int batch_size, int heads, int seq_length, int dim_head, float scale);
void linear_final_attn(float* attention_output, float* out_weights, float* out_bias, float* linear_output, int batch_size, int seq_length, int inner_dim, int dim);
