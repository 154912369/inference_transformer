#pragma once
#include <unistd.h>

class SynChronize{
    int* mem;
    int size;

    __pid_t* _cpid;
    int _cpid_size;
    bool _is_parent;
    public:
    SynChronize(int* shaired_memory, __pid_t* cpid, int cpid_size,bool  _is_parent);
    int get_length();
    void set_length(int length);
    int get_synchronize_length();
    void stop_subprocess();
};