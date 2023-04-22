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
    if(result.get_shape().size()==2){
        transpose_in[0] = input.get_shape()[1];
        transpose_in[1] = input.get_shape()[0];
        transpose_in[2] = input.get_shape()[2];
    }else{
        transpose_in[0] = result.get_shape()[0];
        transpose_in[1] = result.get_shape()[1];
        transpose_in[2] = result.get_shape()[2];
    }
    reshape_copy<<<1,1024>>>(input.get(), result.get(),transpose_in[0],transpose_in[1],transpose_in[2]);
}

__global__ void expf(float *in, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (length-1)/(gridDim.x * blockDim.x )+1;
    int start = tid*size;
    for(int i=0;i<size;i++&&start<length){
        in[start]=expf(in[start]);
        start+=1;
    }  
}
void expf(TensorCUDA& input){
    expf<<<1,1024>>>(input.get(),input.get_size()); 
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
}