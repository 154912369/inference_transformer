#include "tensor/tensor_dynamic_cuda.h"

TensorDynamicCUDA::TensorDynamicCUDA(){

 }

TensorDynamicCUDA::TensorDynamicCUDA(const std::vector<int>& shape){
    _shape = shape;
    _value_size =1;
    for(int i =0;i<shape.size();i++){
        _value_size *= shape[i];
    }
    _real_value_size = 1;
    while (_real_value_size*0.75<_value_size){
        _real_value_size *= 2;
    }
    
    cudaError_t cudaStatus = cudaMalloc(&_device_value_ptr, _real_value_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    cudaStatus = cudaMemset(_device_value_ptr, 0,  _real_value_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
}
TensorDynamicCUDA::~TensorDynamicCUDA(){}
void TensorDynamicCUDA::concat(TensorCUDA& other){
    bool remalloc=false;
    if(_value_size+other.get_size()>_real_value_size*0.75){
        _real_value_size *=2;
        remalloc=true;
    }
    if(remalloc){
        float* new_device_value_ptr;
        cudaError_t cudaStatus = cudaMalloc(&new_device_value_ptr, _real_value_size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
            // 进行错误处理
        }
        cudaStatus = cudaMemset(_device_value_ptr, 0,  _real_value_size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            printf("cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
            // 进行错误处理
        }
        cudaStatus = cudaMemcpy(new_device_value_ptr, _device_value_ptr, _value_size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (cudaStatus != cudaSuccess) {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
            // 进行错误处理
        }
        cudaFree(_device_value_ptr);
        _device_value_ptr = new_device_value_ptr;
    }
    for(int i =1;i<_shape.size();i++){
        if(_shape[i]!=other.get_shape()[i]){
            printf("shape %d : %d, %d is not equal\n", i, _shape[i], other.get_shape()[i]);
        }
    }
    cudaError_t cudaStatus = cudaStatus = cudaMemcpy(_device_value_ptr+_value_size, other.get(), other.get_size() * sizeof(float),  cudaMemcpyDeviceToDevice);

    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    _shape[0] = _shape[0] + other.get_shape()[0];
}