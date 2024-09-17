// Author: Aness Al-Qawlaq 
// Date: 21/2/2024
// University College Dublin


#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

extern const float pos_embedding_weights_layer[324];
extern const float cls_token_weights_layer[12];
extern const float to_patch_embedding_1_weight_weights_layer[192];
extern const float to_patch_embedding_1_bias_weights_layer[12];
extern const float transformer_layers_0_0_norm_weight_weights_postnorm[12];
extern const float transformer_layers_0_0_norm_bias_weights_postnorm[12];
extern const float transformer_layers_0_0_fn_to_qkv_weight_weights_attention[288];
extern const float transformer_layers_0_0_fn_to_out_0_weight_weights_attention[96];
extern const float transformer_layers_0_0_fn_to_out_0_bias_weights_attention[12];
extern const float transformer_layers_0_1_norm_weight_weights_postnorm[12];
extern const float transformer_layers_0_1_norm_bias_weights_postnorm[12];
extern const float transformer_layers_0_1_fn_net_0_weight_weights_feedforward[288];
extern const float transformer_layers_0_1_fn_net_0_bias_weights_feedforward[24];
extern const float transformer_layers_0_1_fn_net_3_weight_weights_feedforward[288];
extern const float transformer_layers_0_1_fn_net_3_bias_weights_feedforward[12];
extern const float mlp_head_0_weight_weights_layer[12];
extern const float mlp_head_0_bias_weights_layer[12];
extern const float mlp_head_1_weight_weights_layer[24];
extern const float mlp_head_1_bias_weights_layer[2];
#endif // MODEL_WEIGHTS_H
