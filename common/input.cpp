#include "common/input.h"
#include <iostream>

Input::Input(std::string model_path){
    this->processor.Load(model_path);
}

bool Input::get_input(std::vector<std::string>& pieces){
    std::string input_str;
    std::getline(std::cin, input_str);
    if(input_str.size()==1&&input_str[0]=='n'){
        pieces.clear();
       pieces.push_back("n");
       first = true;
       return first;
    }
    processor.Encode(input_str, &pieces);
    bool tmp = first;
    if(first){
        first = false;
    }
    return tmp;

}
    