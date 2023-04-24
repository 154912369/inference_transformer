#include "common/cache.h"

void KeyValueCache::add_cache(TensorDynamicCUDA* key, TensorDynamicCUDA* value){
    if(key_list.size() == 0){
        _cache_step = key->get_shape()[0];
    }
    key_list.push_back(std::unique_ptr<TensorDynamicCUDA>(key));
    value_list.push_back(std::unique_ptr<TensorDynamicCUDA>(value));
}

TensorDynamicCUDA* KeyValueCache::get_cache_key(int i){
    return key_list[i].get();
}
TensorDynamicCUDA* KeyValueCache::get_cache_value(int i){
    return value_list[i].get();
}
int KeyValueCache::get_step(){
    return _cache_step;
}


TensorDynamicCUDA* KeyValueCache::incr_key_cache(TensorCUDA& step, int layer_size){
    TensorDynamicCUDA* key = key_list[layer_size].get();
    key->concat(step);
    return key;

}

TensorDynamicCUDA* KeyValueCache::incr_value_cache(TensorCUDA& step, int layer_size){
    TensorDynamicCUDA* key = value_list[layer_size].get();
    key->concat(step);
    return key;
}
