#include "tensor/tensor_cpu.h"
#include <string.h> 
TensorCPU::TensorCPU(const std::string& file_path){
         std::size_t found = file_path.rfind("/");
        if(found==std::string::npos){
            _name = file_path; 
        }else{
            _name = file_path.substr(found+1);
        }
        printf("%s\n",_name.data());
        std::ifstream input(file_path.c_str(), std::ios::binary);
        int shape_size = 0;
        input.read(reinterpret_cast<char*>(&shape_size), sizeof(int));
        int shape_length = 0;
        _value_size = 1;
        for(int i=0;i<shape_size;i++){
            input.read(reinterpret_cast<char*>(&shape_length), sizeof(int));
            _value_size *=  shape_length;
            _shape.push_back(shape_length);
        }

        
       _value = new float[_value_size];
        input.read(reinterpret_cast<char*>(_value), _value_size*sizeof(float));
        input.close();
} 

TensorCPU::TensorCPU(){}

TensorCPU::TensorCPU(const std::vector<int>& shape):_shape(shape){
    _value_size =1;
    for(int i =0;i<_shape.size();i++){
        _value_size *= _shape[i];
    }
    _value  =new float[_value_size];
    memset(_value, 0, _value_size*sizeof(float));
}

TensorCPU::~TensorCPU(){
    if(_value != NULL){
        delete[] _value;
    }
}
const std::vector<int>& TensorCPU::get_shape() const{
    return _shape;
}

int  TensorCPU::get_size() const{
    return _value_size;
}

float* TensorCPU::get() const{
    return _value;
}

const std::string& TensorCPU::get_name() const{
    return _name;
}

void TensorCPU::print() const{
    printf("shape is:");
    for(int i =0;i<_shape.size();i++){
        printf("%d ",_shape[i]);
    }
    printf("\n");
    int second_size = _value_size/_shape[0];
    for(int i=0;i<_shape[0];i++){
        for(int j=0;j<second_size ;j++){
            printf("%.3f ",_value[i*second_size+j]);
        }
         printf("\n");
    }

}