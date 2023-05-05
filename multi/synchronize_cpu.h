#pragma once
#include <unistd.h>

class SynChronize{
    int* mem;
    int size;
    int input_size = 0;
    int reset_size = 0;

    __pid_t* _cpid;
    int _cpid_size;
    bool _is_parent;
    public:
    SynChronize(int* shaired_memory, __pid_t* cpid, int cpid_size,bool  _is_parent);
    int get_length();
    void set_length(int length);
    int get_synchronize_length();
    void stop_subprocess();
    bool wait_for_input();
    void sync_get_input();

     void reset_cache();

};