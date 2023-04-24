#include "tensor/tensor_cuda.h"
class TensorDynamicCUDA: public TensorCUDA{
    private:
        int _real_value_size = 0;

    public:
        TensorDynamicCUDA();
        TensorDynamicCUDA(const std::vector<int>& shape);
        ~TensorDynamicCUDA();
        void concat(TensorCUDA& other);

};