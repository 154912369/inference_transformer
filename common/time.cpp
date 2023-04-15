#include <chrono>
class Timer{
  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  public:
    Timer() { 
        _start = std::chrono::high_resolution_clock::now();
    }
    void start(){
        _start = std::chrono::high_resolution_clock::now();
    }
    int64_t escapse(){
       std::chrono::duration<int64_t, std::micro> diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - _start);
       return diff.count();
    }
};