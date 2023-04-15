#pragma once
#include "tensor/tensor_cpu.h"
class TensorCUDA{
private:
    float* _device_value_ptr = NULL;
    std::vector<int> _shape;
    int _value_size = 0;
    std::string _name;
public:
    TensorCUDA(const TensorCPU& _tensor_cpu);
    TensorCUDA(const std::vector<int>& shape);
    ~TensorCUDA();

    const std::vector<int>& get_shape() const;
    int get_size() const;
    const std::string& get_name() const;
    float* get() const;
    void print() const;
    void cpu(TensorCPU& tensor_cpu);
};

class TensorIntCUDA{
private:
    int* _device_value_ptr = NULL;
    std::vector<int> _shape;
    int _value_size = 0;
    std::string _name;
public:
    TensorIntCUDA(int* input, int input_length);
    ~TensorIntCUDA();

    const std::vector<int>& get_shape() const;
    int get_size() const;
    const std::string& get_name() const;
    int* get() const;
    void print() const;
};

 