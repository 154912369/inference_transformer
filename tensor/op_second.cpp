#include "tensor/op_second.h"
#include "cuda_op/rope.h"

RoPeOP::RoPeOP(int block_x, int thread_x){
    _block_x= block_x;
    _thread_x= thread_x;
}
void RoPeOP::process(const TensorCUDA& input1, const TensorCUDA& input2, TensorCUDA& output){
    rope(input1,input2,output,_block_x,_thread_x);
}