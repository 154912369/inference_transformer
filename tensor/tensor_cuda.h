#pragma once
#include "tensor/tensor_cpu.h"
class TensorCUDA{
protected:
    float* _device_value_ptr = NULL;
    std::vector<int> _shape;
    int _value_size = 0;
    std::string _name;
public:
    TensorCUDA();
    TensorCUDA(const TensorCPU& _tensor_cpu);
    TensorCUDA(const std::vector<int>& shape);
    ~TensorCUDA();

    const std::vector<int>& get_shape() const;
    int get_size() const;
    float* get() const;
    const std::string& get_name() const;
    void print(int offset_i=0, int offset_j=0) const;
    void cpu(TensorCPU& tensor_cpu) const;
    bool equal(const TensorCUDA& _tensor);
    void reshape(const std::vector<int>& shape);
    void reshape_copy(TensorCUDA& result);
    void save(std::string name);
};

class BatchTensorCUDA{
private:
    float** _device_value_ptr = NULL;
    std::vector<int> _shape;
    int _value_size = 0;
    std::string _name;
public:
    BatchTensorCUDA(const TensorCUDA& _tensor_cpu);
    ~BatchTensorCUDA();
    float** get() const;
    void print(int index, int offset_i=0, int offset_j=0) const;

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

 