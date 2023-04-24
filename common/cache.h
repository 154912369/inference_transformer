
#pragma once
#include <vector>
#include "tensor/tensor_cuda.h"
#include <memory>
#include "tensor/tensor_dynamic_cuda.h"
class KeyValueCache{
    int _cache_step = 0;
    std::vector<std::unique_ptr<TensorDynamicCUDA>> key_list;
    std::vector<std::unique_ptr<TensorDynamicCUDA>> value_list;

public:
    void add_cache(TensorDynamicCUDA* key, TensorDynamicCUDA* value);
    TensorDynamicCUDA* get_cache_key(int i);
    TensorDynamicCUDA* get_cache_value(int i);
    int get_step();
    TensorDynamicCUDA* incr_key_cache(TensorCUDA& step, int layer_size);
    TensorDynamicCUDA* incr_value_cache(TensorCUDA& step, int layer_size);

};