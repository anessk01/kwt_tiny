// Author: Aness Al-Qawlaq 
// Date: 21/2/2024
// University College Dublin

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
#include "model_weights_t.h"
#include "in_ds.h"

#define INPUT_RES_1 16
#define INPUT_RES_2 26
#define PATCH_RES_1 16
#define PATCH_RES_2 1
#define DIM 12
#define DEPTH 1
#define HEADS 1
#define MLP_DIM 24
#define DIM_HEAD 8
#define SEQLEN 27
#define MEM1_SIZE SEQLEN*MLP_DIM
#define MEM2_SIZE SEQLEN*DIM_HEAD*3


float global_membank_1[MEM1_SIZE];
float global_membank_2[MEM2_SIZE];

void manual_alloc(float** local_var, int global_bank, int size) {
    float* selected_bank = NULL;

    // Select the appropriate global memory bank
    switch (global_bank) {
        case 1:
            selected_bank = global_membank_1;
            if (size > MEM1_SIZE) {
                exit(-1);
            }
            break;
        case 2:
            selected_bank = global_membank_2;
            if (size > MEM2_SIZE) {
                exit(-1);
            }
            break;
        default:
            exit(-1);
    }

    // Assign the selected memory bank to the local variable
    *local_var = selected_bank;
}

// local_var is q or k or v, partition is one of (1,2,3), size is size of q and k and v 
void manual_alloc_qkv(float** local_var, int global_bank, int partition, int size){
    float* selected_bank = NULL;
    

    // Select the appropriate global memory bank
    switch (global_bank) {
        case 1:
            selected_bank = global_membank_1;
            if (size*3 > SEQLEN * MEM1_SIZE) {
                exit(-1);
            }
            break;
        case 2:
            selected_bank = global_membank_2;
            if (size*3 > MEM2_SIZE) {
                exit(-1);
            }
            break;
        default:
            exit(-1);
    }

    switch (partition) {
        case 1:
            *local_var = selected_bank;
            break;
        case 2:
            *local_var = selected_bank+size;
            break;
        case 3:
            *local_var = selected_bank+size*2;
            break;
        default:
            exit(-1);
    }
}
// example usage: 
// instead of:    this->weight1 = (float*)malloc(64 * 12 * sizeof(float));
// I have:  manual_alloc(this->weight1, global_membank_1, 64*12)

typedef struct {
    float *qkv_weights; // Dynamic array for combined query, key, value projection weights
    float *out_weights; // Dynamic array for output projection weights
    float *out_bias; // Dynamic array for output projection bias
    int dim; // Dimension of input
    int heads; // Number of attention heads
    int dim_head; // Dimension per attention head
    float scale; // Scale factor
    float *qkv_output; // Dynamic array for combined query, key, value output
    float *q; // Dynamic array for queries
    float *k; // Dynamic array for keys
    float *v; // Dynamic array for values
    float attention_scores[SEQLEN*SEQLEN]; // Dynamic array for attention scores
    float *output; // Dynamic array for final output
} Attention;

typedef struct {
    float *gamma; // Dynamic array for Scale parameter for LayerNorm
    float *beta; // Dynamic array for Shift parameter for LayerNorm
    int dim; // Dimension for LayerNorm
    void (*fn)(Attention*, float*, float*, int, int, int);
} PostNorm_attn;

typedef struct {
    float *weight1; // Dynamic array for the first linear layer
    float *bias1; // Dynamic array for the first linear layer bias
    float *weight2; // Dynamic array for the second linear layer
    float *bias2; // Dynamic array for the second linear layer bias
    int input_dim; // Dimension of the input to the first linear layer
    int hidden_dim; // Dimension of the hidden layer
} FeedForward;

typedef struct {
    float *gamma; // Dynamic array for Scale parameter for LayerNorm
    float *beta; // Dynamic array for Shift parameter for LayerNorm
    int dim; // Dimension for LayerNorm
    void (*fn)(FeedForward*, float*, float*, int);
} PostNorm;

typedef struct {
    Attention attention;
    PostNorm post_norm;
    PostNorm_attn post_norm_attn;
    FeedForward ff;
} TransformerLayer;

typedef struct {
    TransformerLayer layers[1]; // Pointer to an array of Transformer layers
    int depth; // Number of layers
    int dim; // Input and output dimension
} Transformer;

typedef struct {
    float *patch_embedding_linear_weight;
    float *patch_embedding_linear_bias;
    float *cls_token;
    float *pos_embeddings;
    float *mlp_norm_gamma;
    float *mlp_norm_beta;
    float *linear_weights;
    float *linear_bias;
} KWT;

// Function prototypes for initialization
void initializePostNorm(PostNorm *postNorm, FeedForward* ff);
void initializePostNorm_attn(PostNorm_attn *postNorm, Attention* attention);
void initializeFeedForward(FeedForward *ff, int dim, int mlp_dim);
void initializeAttention(Attention *attention, int dim, int heads, int dim_head);
void initializeTransformerLayer(TransformerLayer *layer, int dim, int heads, int dim_head, int mlp_dim);
void initializeTransformer(Transformer *transformer, int dim, int heads, int dim_head, int mlp_dim, int depth);
void initializeKWT(KWT* model);

// Initialization functions for various structs

void initializeAttention(Attention *attention, int dim, int heads, int dim_head) {
    attention->dim = dim;
    attention->heads = heads;
    attention->dim_head=dim_head;
    attention->scale = 1.0f / sqrtf((float)dim_head);
}

void initializeFeedForward(FeedForward *ff, int dim, int mlp_dim) {
    ff->input_dim = dim;
    ff->hidden_dim = mlp_dim;
}

void initializeTransformerLayer(TransformerLayer *layer, int dim, int heads, int dim_head, int mlp_dim) {
    initializeFeedForward(&layer->ff, dim, mlp_dim);
    initializeAttention(&layer->attention, dim, heads, dim_head);
    initializePostNorm(&layer->post_norm, &layer->ff);
    initializePostNorm_attn(&layer->post_norm_attn, &layer->attention);
}

void initializeTransformer(Transformer *transformer, int dim, int heads, int dim_head, int mlp_dim, int depth) {
    initializeTransformerLayer(&transformer->layers[0], dim, heads, dim_head, mlp_dim);
    transformer->dim = dim;
    transformer->depth = depth;
}

void initializeKWT(KWT* model) {
    model->cls_token = (float *) cls_token_weights_layer;
    model->mlp_norm_gamma = (float *) mlp_head_0_weight_weights_layer;
    model->mlp_norm_beta = (float *) mlp_head_0_bias_weights_layer;
    model->linear_weights = (float *) mlp_head_1_weight_weights_layer;
    model->linear_bias = (float *) mlp_head_1_bias_weights_layer;
}

// Processing functions

void forwardPostNorm(PostNorm* this, FeedForward* ff, float* x, float* output, int batch_size, int seq_length, int dim) {
    int input_size = batch_size * seq_length * dim;
    float* fn_output;
    manual_alloc(&fn_output, 2, input_size);
    
    this->fn(ff, x, fn_output, seq_length);

    // feedforward postnorm init
    this->gamma = (float *) transformer_layers_0_1_norm_weight_weights_postnorm;
    this->beta = (float *) transformer_layers_0_1_norm_bias_weights_postnorm;
    layerNorm(fn_output, output, this->gamma, this->beta, this->dim, input_size);
    // free(fn_output);
}

void forwardPostNorm_attn(PostNorm_attn* this, Attention* attention, float* x, float* output, int batch_size, int seq_length, int dim) {
    int input_size = batch_size * seq_length * dim;
    float* fn_output;
    manual_alloc(&fn_output, 2, input_size);
    
    this->fn(attention, x, fn_output, batch_size, seq_length, dim);

    this->gamma = (float *) transformer_layers_0_0_norm_weight_weights_postnorm;
    this->beta = (float *) transformer_layers_0_0_norm_bias_weights_postnorm;
    layerNorm(fn_output, output, this->gamma, this->beta, this->dim, input_size);
    // free(fn_output);
}

void forwardFeedForward(FeedForward* this, float* x, float* output, int seq_len) {
    int hidden_output_size = seq_len * this->hidden_dim;
    float* hidden_output;

    manual_alloc(&hidden_output, 1, hidden_output_size);

    this->weight1 = (float *) transformer_layers_0_1_fn_net_0_weight_weights_feedforward;
    this->bias1 = (float *) transformer_layers_0_1_fn_net_0_bias_weights_feedforward;
    linear(this->weight1, this->bias1, x, hidden_output, seq_len, this->input_dim, this->hidden_dim, this->hidden_dim);

    gelu(hidden_output, hidden_output_size);
    
    this->weight2 = (float *) transformer_layers_0_1_fn_net_3_weight_weights_feedforward;
    this->bias2 = (float *) transformer_layers_0_1_fn_net_3_bias_weights_feedforward;
    // Copy the values to the struct
    linear(this->weight2, this->bias2, hidden_output, output, seq_len, this->hidden_dim, this->input_dim, this->input_dim);
    
    // free(hidden_output);
}

void forwardAttention(Attention* this, float* x, float* output, int batch_size, int seq_length, int dim) {
    int inner_dim = this->dim_head * this->heads;
    float* qkv_output_temp;
    manual_alloc(&qkv_output_temp, 2, seq_length * inner_dim * 3);

    this -> qkv_weights = (float *) transformer_layers_0_0_fn_to_qkv_weight_weights_attention;
    matrixMultiply(x, this->qkv_weights, qkv_output_temp, batch_size * seq_length, this->dim, inner_dim * 3);

    manual_alloc(&this->qkv_output, 1, SEQLEN*DIM_HEAD*3);

    chunk_qkv(qkv_output_temp, this->qkv_output, seq_length, inner_dim * 3);
    // free(qkv_output_temp);
    
    manual_alloc_qkv(&this->q, 2, 1, SEQLEN*DIM_HEAD);
    manual_alloc_qkv(&this->k, 2, 2, SEQLEN*DIM_HEAD);
    manual_alloc_qkv(&this->v, 2, 3, SEQLEN*DIM_HEAD);

    splitIntoQKV(this->qkv_output, this->q, this->k, this->v, batch_size, seq_length, this->heads, this->dim_head);
    // free(this->qkv_output);
    
    manual_alloc(&this->output, 1, SEQLEN*MLP_DIM);

    // STACK SIZE ATTENTION SCORES 27x27x2 
    scaledDotProductAttention(this->q, this->k, this->v, this->attention_scores, this->output, batch_size, this->heads, seq_length, this->dim_head, this->scale);
    // free(this->q);
    // free(this->k);
    // free(this->v);
    // free(this->attention_scores);

    this->out_weights = (float *) transformer_layers_0_0_fn_to_out_0_weight_weights_attention;
    this->out_bias = (float *) transformer_layers_0_0_fn_to_out_0_bias_weights_attention;
    linear_final_attn(this->output, this->out_weights, this->out_bias, output, batch_size, seq_length, inner_dim, dim);
    // free(this->output);
}

void increment_output(float* x, float* output, int seq_length, int dim) {
    for (int i = 0; i < seq_length * dim; i++) {
        output[i] += x[i];

        // save new output in x so that it can be referred to after feedforward
        x[i] = output[i];
    }
}

void forwardTransformerLayer(TransformerLayer* layer, float* x, float* output, int batch_size, int seq_length, int dim) {
    forwardPostNorm_attn(&layer->post_norm_attn, &layer->attention, x, output, batch_size, seq_length, dim);
    increment_output(x, output, seq_length, dim);
    forwardPostNorm(&layer->post_norm, &layer->ff, output, output, batch_size, seq_length, dim);
    increment_output(x, output, seq_length, dim);
}

void forwardTransformer(Transformer* model, float* x, float* output, int batch_size, int seq_length) {
    for (int i = 0; i < model->depth; i++) {
        forwardTransformerLayer(&model->layers[i], x, output, batch_size, seq_length, model->dim);
        // Update x for the next layer
        x = output;
    }
    // free(model->layers);
}

void forwardKWT(KWT* this, Transformer* transformer, float* x, float* output) {
    // float* intermediate_1 = (float*)malloc(SEQLEN * DIM * sizeof(float));
    // float* intermediate_2 = (float*)malloc(SEQLEN * DIM * sizeof(float));
    float intermediate_1[SEQLEN * DIM];
    float intermediate_2[SEQLEN * DIM];
    // STACK SIZE: 27x12x2x2


    this->patch_embedding_linear_weight = (float*) to_patch_embedding_1_weight_weights_layer;
    this->patch_embedding_linear_bias = (float*) to_patch_embedding_1_bias_weights_layer;
    linear(this->patch_embedding_linear_weight, this->patch_embedding_linear_bias, x, intermediate_1, INPUT_RES_2, INPUT_RES_1, DIM, DIM);

    concat_array(this->cls_token, intermediate_1, DIM, (SEQLEN - 1) * DIM, intermediate_2);

    this->pos_embeddings = (float*) pos_embedding_weights_layer;
    add_overwrite(intermediate_2, this->pos_embeddings, SEQLEN * DIM);

    forwardTransformer(transformer, intermediate_2, intermediate_1, 1, SEQLEN);
    layerNorm(intermediate_1, intermediate_2, this->mlp_norm_gamma, this->mlp_norm_beta, DIM, DIM * SEQLEN);
    linear(this->linear_weights, this->linear_bias, intermediate_2, output, 1, DIM, 2, DIM);

    // free(intermediate_1);
    // free(intermediate_2);
}

void initializePostNorm(PostNorm *postNorm, FeedForward* ff) {
    postNorm->dim = DIM;

    // Set the function pointer to forwardFeedForward
    postNorm->fn = (void (*)(FeedForward*, float*, float*, int)) forwardFeedForward;
}

void initializePostNorm_attn(PostNorm_attn *postNorm, Attention* attention) {
    postNorm->dim = DIM;

    // Set the function pointer to forwardAttention
    postNorm->fn = (void (*)(Attention*, float*, float*, int, int, int)) forwardAttention;
}


int main() {
    // Initialize the Transformer model
    Transformer transformer;
    initializeTransformer(&transformer, DIM, HEADS, DIM_HEAD, MLP_DIM, DEPTH);

    // Define the output tensor
    float output_tensor[2];

    KWT model;
    initializeKWT(&model);

    // Assuming input_tensor_transposed is defined and initialized elsewhere
    forwardKWT(&model, &transformer, input_tensor_transposed, output_tensor);

    // Print the output tensor
    printf("Original Model Output:\n");
    for (int j = 0; j < 2; j++) {
        printf("%f ", output_tensor[j]);
    }
    printf("\n");
    return 0;
}

