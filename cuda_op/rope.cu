#include "cuda_op/rope.h"


// rope_q[s,m,2*i]=pos_embedding[s,2*i]*origin_q[s,m,2*i]-pos_embedding[s,2*i+1]*origin_q[s,m,2*i+1]
// rope_q[s,m,2*i+1]=pos_embedding[s,2*i]*origin_q[s,m,2*i+1]+pos_embedding[s,2*i+1]*origin_q[s,m,2*i]
__global__ void rope(float* origin_q,float* pos_embedding,float* rope_q,int length,int head_count, int hidden_size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int  size = (head_count*hidden_size)/(blockDim.x*gridDim.x);
    int m = (tid*size)/hidden_size;
    int start_index = tid%(hidden_size/size); //表示2*i;
    float first = 0;
    float second = 0;
    for(int s=0;s<length;s++){
        for(int i=start_index*size/2; 2*i+1<start_index*size+size;i++){
            first = origin_q[s*head_count*hidden_size+m*hidden_size+2*i];
            second = origin_q[s*head_count*hidden_size+m*hidden_size+2*i+1];
            rope_q[s*head_count*hidden_size+m*hidden_size+2*i]=pos_embedding[s*hidden_size+2*i]*first-pos_embedding[s*hidden_size+2*i+1]*second;
            rope_q[s*head_count*hidden_size+m*hidden_size+2*i+1]=pos_embedding[s*hidden_size+2*i]*second+pos_embedding[s*hidden_size+2*i+1]*first;
                    }
    }
   
     
}

void 

rope(const TensorCUDA& input1,
                const TensorCUDA& input2,
                TensorCUDA& result,
                int block_x,  int thread_x){
    int hidden_size = input2.get_shape()[1]; 
    int length = input2.get_shape()[0]; 
    int head_count = input1.get_shape()[1]/ hidden_size; 
    rope<<<1,1024>>>(input1.get(), input2.get(), result.get(),length, head_count,hidden_size);
}