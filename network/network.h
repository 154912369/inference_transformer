#include   "tensor/tensor_cuda.h"
#include "tensor/op.h"
#include "tensor/op_second.h"
#include <memory>
class Transformer{
    EmbeddingOP* _word_embedding = NULL;
    EmbeddingOP* _sent_embedding = NULL;
    EmbeddingOP* _role_embedding = NULL;
    EmbeddingOP* _pos_embedding = NULL;
    LayerNormlizeOP* _post_encoder_layer_norm = NULL;
    MatAddOP* _mat_add_op = NULL;
    MatMulOP* _matmulop  = NULL;
    RoPeOP* _rope_op = NULL;
    int _layer_size = 0;

    std::vector<std::unique_ptr<LayerNormlizeOP>> _reduce_2d_sum_op;
    std::vector<std::unique_ptr<LayerNormlizeOP>> _pre_ffn_layer_norm; 

    std::vector<std::unique_ptr<TensorCUDA>>  _query_w;
    std::vector<std::unique_ptr<TensorCUDA>>  _key_w;
    std::vector<std::unique_ptr<TensorCUDA>>  _value_w;
    std::vector<std::unique_ptr<TensorCUDA>>  _output_w;
    std::vector<std::unique_ptr<TensorCUDA>>  _ffn0_weight;
    std::vector<std::unique_ptr<TensorCUDA>>  _ffn1_weight;
    
    



    public:
        Transformer(const std::string& path, int layer_size);
        ~Transformer();
        TensorCUDA* get_embedding_out(int* token_type_list,
                    int* role_ids_list,
                    int* sent_ids_list,
                    int length);
        TensorCUDA* get_pos_embedding_out(int* token_type_list,
                    int length);
        void init_root(const std::string& model_path);
        void get_q_k_v(TensorCUDA& tensor,TensorCUDA& pos_type_embedding,  TensorCUDA& q, TensorCUDA& k, TensorCUDA& v, int layer_index);

        void get_attention_output(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index);

        void encoder_layer(TensorCUDA& tensor,TensorCUDA& pos_type_embedding, int layer_index);
        void encode(TensorCUDA& tensor,TensorCUDA& pos_type_embedding);

};