#include "tensor/tensor_cuda.h"
#include "cuda_op/common.h"
#include  <fstream>

TensorCUDA::TensorCUDA(){}

TensorCUDA::TensorCUDA(const TensorCPU& _tensor_cpu){
    cudaError_t cudaStatus;
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

void TensorCUDA::save(std::string name) const{

    std::ofstream file("/data1/renweijie/baidu/dialogue/nlg-paddle-inference/transfer_output/"+name, std::ios::binary);
    int size = _shape.size();
    file.write ((char *)&size, sizeof(int));
    for(int i=0;i<_shape.size();i++){
        file.write ((char *)&(_shape[i]), sizeof(int));
    }
    float* tmp_value = new float[_value_size];
    cudaError_t cudaStatus = cudaMemcpy( tmp_value, _device_value_ptr, _value_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    file.write ((char *) tmp_value, _value_size * sizeof(float));
    delete[] tmp_value;
    file.close (); 
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

void TensorCUDA::cpu(TensorCPU& tensor_cpu) const{
    cudaMemcpy( tensor_cpu.get(), _device_value_ptr, _value_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void TensorCUDA::print(int offset_i, int offset_j) const{
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
    int first_size = _shape[0];
    if(_shape.size()==1){
        first_size= first_size <20 ? first_size:20;
        for(int i=0;i<first_size;i++){
            printf("%.5f ",tmp_value[i]);
        }
        return;
    }
    first_size= first_size <5 ? first_size:5;
    int origin_second_size = _value_size/_shape[0];
    int second_size = origin_second_size <10 ?origin_second_size:10;
    for(int i= offset_i;i<first_size+ offset_i;i++){
        for(int j=offset_j;j<second_size+offset_j;j++){
            printf("%.5f(%d,%d) ",tmp_value[i*origin_second_size +j],i,j);
        }
         printf("\n");
    }
    delete[] tmp_value;

}



bool TensorCUDA::equal(const TensorCUDA& tensor){
    std::vector<int> shape = tensor.get_shape();
    if(shape.size() != _shape.size()){
        printf("shape size is %d,%d", shape.size(), _shape.size());
        return false;
    }
    for(int i =0; i<_shape.size();i++){
        if(shape[i]!=_shape[i]){
            printf("shape  is  not equal %d: %d,%d", i, shape[i], _shape[i]);
            return false;
        }
    }

    float* tmp_value = new float[_value_size];
    cudaError_t cudaStatus = cudaMemcpy( tmp_value, _device_value_ptr, _value_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }

    float* tmp_value_1 = new float[_value_size];
    cudaStatus = cudaMemcpy( tmp_value_1, tensor.get(), _value_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    for(int i =0;i<_value_size;i++){
        if(abs(tmp_value[i]-tmp_value_1[i])>0.056&&\
            abs(tmp_value[i]-tmp_value_1[i])/(abs(tmp_value[i])+abs(tmp_value_1[i]))>0.1){
            printf("diff is %d : %f, %f, %f, %f\n", i, tmp_value[i],tmp_value_1[i],abs(tmp_value[i]-tmp_value_1[i]), abs(tmp_value[i]-tmp_value_1[i])/(abs(tmp_value[i])+abs(tmp_value_1[i])));
            delete[] tmp_value;
            delete[] tmp_value_1;
            return false;
        }
    }
    delete[] tmp_value;
    delete[] tmp_value_1;
    return true;

}
void  TensorCUDA::reshape(const std::vector<int>& shape){
    int size = 1;
    for(int i =0;i<shape.size();i++){
        size*=shape[i];
    }
    if(size==_value_size ){
        _shape = shape;
    }else{
        printf("this size is %d ,new size is %d\n, skip reshape", _value_size, size);
    }
}
void TensorCUDA::reshape_copy(TensorCUDA& result){
    reshape_copy_2d(*this, result );
}

BatchTensorCUDA::BatchTensorCUDA(const TensorCUDA& _tensor_cpu):_shape(_tensor_cpu.get_shape()){
    _value_size = _tensor_cpu.get_size();
    cudaError_t cudaStatus = cudaMalloc(&_device_value_ptr, _tensor_cpu.get_shape()[0]* sizeof(float*));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    float* input[ _tensor_cpu.get_shape()[0]];
    int step_size = _tensor_cpu.get_shape()[1]*_tensor_cpu.get_shape()[2];
    float* start = _tensor_cpu.get();
    for(int i=0;i<_tensor_cpu.get_shape()[0];i++){
        input[i]=start;
        start+= step_size;
    }
    cudaStatus = cudaMemcpy(_device_value_ptr, input,  _tensor_cpu.get_shape()[0]* sizeof(float*), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
}

BatchTensorCUDA::~BatchTensorCUDA(){
    if(_device_value_ptr){
        // cudaError_t cudaStatus = cudaFree(_device_value_ptr);
        // if (cudaStatus != cudaSuccess) {
        //     printf("batch cudaFree failed: %s %p\n", cudaGetErrorString(cudaStatus), _device_value_ptr);
        //     // 进行错误处理
        // }
    }
}
float** BatchTensorCUDA::get() const{
    return _device_value_ptr;
}

void BatchTensorCUDA::print(int index, int offset_i, int offset_j) const{
    float** tmp_value_index = new float*[_shape[0]];
    float* tmp_value = new float[_value_size/_shape[0]];
    cudaError_t cudaStatus = cudaMemcpy( tmp_value_index, _device_value_ptr, _shape[0] * sizeof(float*), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    cudaStatus = cudaMemcpy( tmp_value, tmp_value_index[index],_value_size/_shape[0] * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    printf("cuda shape is:");
    for(int i =0;i<_shape.size();i++){
        printf("%d ",_shape[i]);
    }
    printf("\n");
    int first_size = _shape[1];
    if(_shape.size()==1){
        first_size= first_size <20 ? first_size:20;
        for(int i=0;i<first_size;i++){
            printf("%.5f ",tmp_value[i]);
        }
        return;
    }
    first_size= first_size <5 ? first_size:5;
    int origin_second_size = _value_size/_shape[1]/_shape[0];
    int second_size = origin_second_size <10 ?origin_second_size:10;
    for(int i=offset_i;i<first_size+offset_i;i++){
        for(int j=offset_j;j<second_size+offset_j;j++){
            printf("%.5f(%d,%d) ",tmp_value[i*origin_second_size +j],i,j);
        }
         printf("\n");
    }
    delete[] tmp_value;
    delete[] tmp_value_index;

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

