#ifndef UTILS_H
#define UTILS_H

#include <set>
#include <string>

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
};

#endif                