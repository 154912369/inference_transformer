#include "tensor/tensor_cuda.h"

TensorCUDA::TensorCUDA(const TensorCPU& _tensor_cpu){
    int result=-3;
    cudaError_t cudaStatus =cudaRuntimeGetVersion(&result);
    if (cudaStatus != cudaSuccess) {
        printf("cudaError Runtime Version failed: %s %d\n", cudaGetErrorString(cudaStatus), cudaStatus);
        // 进行错误处理
    }
    printf("run time version is : %d\n",result);
    cudaStatus =cudaDriverGetVersion(&result);
    if (cudaStatus != cudaSuccess) {
        printf("cudaError Driver Version failed: %s %d\n", cudaGetErrorString(cudaStatus), cudaStatus);
        // 进行错误处理
    }
    printf("driver time version is : %d\n",result);
    _shape  = _tensor_cpu.get_shape();
    _value_size = _tensor_cpu.get_size();
    cudaStatus = cudaMalloc(&_device_value_ptr, _value_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    cudaStatus = cudaMemcpy(_device_value_ptr, _tensor_cpu.get(), _value_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    
};

TensorCUDA::TensorCUDA(const std::vector<int>& shape):_shape(shape){
    _value_size =1;
    for(int i =0;i<shape.size();i++){
        _value_size *= shape[i];
    }
    cudaError_t cudaStatus = cudaMalloc(&_device_value_ptr, _value_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    cudaStatus = cudaMemset(_device_value_ptr, 0,  _value_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
}
TensorCUDA::~TensorCUDA(){
    if(_device_value_ptr != NULL){
        cudaFree(_device_value_ptr);
    }
}


const std::vector<int>& TensorCUDA::get_shape() const{
    return _shape;
}

int  TensorCUDA::get_size() const{
    return _value_size;
}


const std::string& TensorCUDA::get_name() const{
    return _name;
}

float* TensorCUDA::get() const{
    return _device_value_ptr;
}

void TensorCUDA::cpu(TensorCPU& tensor_cpu){
    cudaMemcpy( tensor_cpu.get(), _device_value_ptr, _value_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void TensorCUDA::print() const{
    float* tmp_value = new float[_value_size];
    cudaError_t cudaStatus = cudaMemcpy( tmp_value, _device_value_ptr, _value_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    printf("cuda shape is:");
    for(int i =0;i<_shape.size();i++){
        printf("%d ",_shape[i]);
    }
    printf("\n");
    int second_size = _value_size/_shape[0];
    for(int i=0;i<_shape[0];i++){
        for(int j=0;j<second_size ;j++){
            printf("%.3f ",tmp_value[i*second_size+j]);
        }
         printf("\n");
    }
    delete[] tmp_value;

}


TensorIntCUDA::TensorIntCUDA(int* input, int input_length){


    _shape.push_back(input_length);
    _value_size = input_length;
    cudaError_t cudaStatus = cudaMalloc(&_device_value_ptr, _value_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    cudaStatus = cudaMemcpy(_device_value_ptr, input, _value_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    
};
TensorIntCUDA::~TensorIntCUDA(){
    if(_device_value_ptr != NULL){
        cudaFree(_device_value_ptr);
    }
}


const std::vector<int>& TensorIntCUDA::get_shape() const{
    return _shape;
}

int  TensorIntCUDA::get_size() const{
    return _value_size;
}


const std::string& TensorIntCUDA::get_name() const{
    return _name;
}

int* TensorIntCUDA::get() const{
    return _device_value_ptr;
}

void TensorIntCUDA::print() const{
    int* tmp_value = new int[_value_size];
    cudaError_t cudaStatus = cudaMemcpy( tmp_value, _device_value_ptr, _value_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    printf("cuda int shape is:");
    for(int i =0;i<_shape.size();i++){
        printf("%d ",_shape[i]);
    }
    printf("\n");
    for(int i=0;i<_shape[0];i++){
        printf("%d ",tmp_value[i]);
    }
    printf("\n");
    delete[] tmp_value;

}