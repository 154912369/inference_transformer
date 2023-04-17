#include   "tensor/tensor_cuda.h"
#include "tensor/op.h"
class Transformer{
    EmbeddingOP* _word_embedding = NULL;
    EmbeddingOP* _sent_embedding = NULL;
    EmbeddingOP* _role_embedding = NULL;
    EmbeddingOP* _pos_embedding = NULL;
    MatAddOP* _mat_add_op = NULL;



    public:
        Transformer(const std::string& path);
        ~Transformer();
        TensorCUDA* get_embedding_out(int* token_type_list,
                    int* role_ids_list,
                    int* sent_ids_list,
                    int length);
        TensorCUDA* get_pos_embedding_out(int* token_type_list,
                    int length);
        void init_root(const std::string& model_path);

};