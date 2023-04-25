#include "common/string_utils.h"
#include <sstream>
#include <algorithm>
std::vector<std::string> split(const std::string& str, char separator){
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (getline(ss, token, separator)) {
        tokens.push_back(token);
    }

    return tokens;
}

void trim(std::string& s){
    s.erase(std::remove(s.begin(), s.end(), '_'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
}