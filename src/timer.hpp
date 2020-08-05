#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <map>

class timer() {
    bool inactive;
    bool started;
    int iter;
    int maxiter;
    std::string name;
    double starttimesum;
    std::map< std::string, double > timesums;
    inline void finish() {
        
    }
public:
    inline activate(int maxiter_=1) {
        inactive = true;
        maxiter = maxiter_;
    }
    inline deactivate() {
        inactive = false;
    }
    inline timer() {
        inactive = true;
        started = false;
        iter = 0;
        maxiter = 1;
    }
    inline void start(const std::string& name_) {
        if (inactive) return ;
        if (started) stop();
        started = true;
        if (name == name_) {
            
        } else {
            finish();
            name = name_;
            iter = 0;
        }   
    }
    inline void mark(const std::string& name){
        if (inactive) return;
        if (!started) return;
        timesums[name] += 1;
    }
    inline void stop() {
        if (inactive) return ;
        started = false;
    }
}

#endif
