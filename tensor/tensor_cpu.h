#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class TensorCPU{
private:
    float* _value = NULL;
    std::vector<int> _shape;
    int _value_size = 0;
    std::string _name;
public:
    TensorCPU();
    TensorCPU(const std::string& file_path);
    TensorCPU(const std::vector<int>& shape);
    ~TensorCPU();

    const std::vector<int>& get_shape() const;
    float* get() const;
    int get_size() const;
    const std::string& get_name() const;
    void print() const;
};


