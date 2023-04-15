#include "tensor/op.h"
#include "common/time.cpp"
#include "unistd.h"
int main(int argc, char* argv[]){
    std::string a="/home/unit/renweijie/nlg-paddle/matrix.bin";
    int block_x = std::atoi(argv[0]), block_y = std::atoi(argv[1]),
        thread_x = std::atoi(argv[2]);
    EmbeddingOP b(a, block_x, block_y, thread_x);


    {
        printf("parallize arg %d %d %d\n", block_x, block_y, thread_x);
        EmbeddingOP b(a, 2, 3, 5);
        int length = 8;
        int index[length];
        for (int i = 0; i < length;i++){
        index[i] = i%3;
        }
        TensorIntCUDA index_cuda(index, length);
        index_cuda.print();
        std::vector<int> shape = {length, b.get_dim()};
        TensorCUDA result(shape);
        Timer time;
        b.process(index_cuda, result);
        printf("time cost %d\n", time.escapse());
        result.print();

        TensorCUDA result1(result.get_shape());
        MatAddOP mat_add_op(2, 3, 5);
        mat_add_op.process(result, result, result1);
        result1.print();
    }
    return 0;
}
