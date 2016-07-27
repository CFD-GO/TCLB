#ifndef UTILS_H
#define UTILS_H

#include <set>
#include <string>

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <errno.h>
#include <assert.h>
#include <sys/stat.h>

inline void stripbare(char * str)
{
	int i = 0, j=0;
	while (str[i])
	{
		str[j] = str[i];
		if (str[j] == '/') j = -1;
		i++;j++;
	}
	str[j] = 0;
	i = 0;
	j = -1;
	while (str[i]) {
		if (str[i] == '.') j =i;
		i++;
	}
	if (j == -1) j = i;
	str[j] = 0;
}

inline int myround(double v) {
	if (v > 0) return v+0.5;
	return v-0.5;
}

class name_set {
	std::set< std::string > myset;
	public:
        typedef std::set<std::string>::iterator iterator;
		inline void add_from_string(std::string in, char separator) {
			std::string::iterator  ts, curr;
			ts = curr = in.begin();
		        for(; curr <= in.end(); curr++ ) {
		        	if( (curr == in.end() || *curr == separator) && curr > ts )
		                	myset.insert( std::string( ts, curr ));
				if( curr == in.end() )
		                	break;
				if( *curr == separator ) ts = curr + 1;
			}
		}
		inline name_set(char * str) {
			add_from_string(str, ',');
		}
		inline name_set() {
		}
		inline bool in(std::string what) {
		        if (myset.count("all") > 0) return true;
			return myset.count(what) > 0;
		}
        inline int size(){
            return myset.size();
        }
        inline std::set< std::string >::iterator begin(){ 
            return myset.begin();
        }   
        inline std::set< std::string >::iterator end(){ 
            return myset.end();
        }
};


// This function creates the full path to a file (creates all the directories)
inline int mkpath(char* file_path_, mode_t mode) {
  char file_path[1024];
  if (file_path_ == NULL) return -1;
  if (*file_path_ == '\0') return 0;
  if (strlen(file_path_) >= STRING_LEN) {
    error("too long path in mkdir: %s\n", file_path_);
    return -1;
  }
  strcpy(file_path,file_path_);
  debug1("mkdir: %s (-p)\n", file_path);
  char* p;
  for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
    *p='\0';
    if (mkdir(file_path, mode)==-1) {
      if (errno!=EEXIST) {
        debug1("mkdir: %s - cannot create\n", file_path);
        *p='/'; return -1;
      } else {
        debug1("mkdir: %s - exists\n", file_path);
      }
    } else {
        debug1("mkdir: %s - created\n", file_path);
    }
    *p='/';
  }
  return 0;
}

inline int mkpath(char* file_path_) {
  return mkpath(file_path_, 0775);
}


#endif                
