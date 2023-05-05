#include "cuda_op/common.h"

__global__ void reshape_copy(float *in, float *out, int first, int second,int third) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (first*second*third-1)/(gridDim.x * blockDim.x )+1;
    int index[3];
    int start = tid*size;
    index[2]=start%third;
    start =start/third;
    index[1]=start%second;
    index[0] = start/second;
    start = tid*size;
    for(int i=0;i<size&&index[0]<first;i++){
        out[start]=in[index[1]*first*third+index[0]*third+index[2]];
        index[2]+=1;
        start+=1;
        if(index[2]==third){
            index[2]=0;
            index[1]+=1;
            if(index[1]==second){
                index[1]=0;
                index[0]+=1;
            }
        }
    }                    
}

void reshape_copy_2d(TensorCUDA& input, TensorCUDA& result){
    int  transpose_in[3];
    if(input.get_shape().size()==3){
        transpose_in[0] = input.get_shape()[1];
        transpose_in[1] = input.get_shape()[0];
        transpose_in[2] = input.get_shape()[2];
    }else{
        transpose_in[0] = result.get_shape()[0];
        transpose_in[1] = result.get_shape()[1];
        transpose_in[2] = result.get_shape()[2];
    }
    reshape_copy<<<1,1024>>>(input.get(), result.get(),transpose_in[0],transpose_in[1],transpose_in[2]);
    cudaError_t cudaStatus =  cudaDeviceSynchronize();
    if (cudaStatus !=  cudaSuccess) {
        printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
}

__global__ void expf(float *in, int length, int head_size,int prelength,bool is_mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (length-1)/(gridDim.x * blockDim.x )+1;
    int start = tid*size;
    for(int i=0;i<size&&start<length;i++){
        int index = (start%head_size)%(head_size/prelength);//最大大于prelength(0,2*prelngth)
        int first = (start%head_size)/(head_size/prelength); //最大为prelength(0,prelngth)
        if(!is_mask|| index+prelength>=first){
            in[start]=expf(in[start]);
        }else{
            in[start]=0;
        }
        start+=1;
    }  
}

bool check_cuda_err1() {
    cudaError_t err = cudaGetLastError();
    if(err == cudaSuccess) {
        return true;
    }
    else {
        printf("Cuda Error: %s \n",cudaGetErrorString(err));
        return false;
    }
}

void expf(TensorCUDA& input, bool is_mask){
    int first = input.get_shape()[1]*input.get_shape()[2];
    int second = input.get_shape()[1];
    expf<<<1,1024>>>(input.get(),input.get_size(),first, second, is_mask); 
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    check_cuda_err1();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
}
void expf(TensorCUDA& input,bool is_mask,int myrank, int ranks){

}
__global__ void gelu(float *in, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (length-1)/(gridDim.x * blockDim.x )+1;
    int start = tid*size;
    for(int i=0;i<size;i++&&start<length){
        in[start]=in[start]*normcdf(in[start]);
        start+=1;
    }  
}
void gelu(TensorCUDA& input){
    gelu<<<1,1024>>>(input.get(),input.get_size()); 
    cudaError_t cudaStatus =  cudaDeviceSynchronize();
    if (cudaStatus !=  cudaSuccess) {
        printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
}

void gelu(TensorCUDA& input,int start, int size){
    gelu<<<1,1024>>>(input.get()+start, size); 
    cudaError_t cudaStatus =  cudaDeviceSynchronize();
    if (cudaStatus !=  cudaSuccess) {
        printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
}
void init_cuda(){
    size_t result;
    cudaDeviceGetLimit(&result, cudaLimitMallocHeapSize);
    printf("cuda device head limit is %d\n", result);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2*1024*1024*1024);
    cudaDeviceGetLimit(&result, cudaLimitMallocHeapSize);
    printf("cuda device head limit is %d\n", result);
    
}