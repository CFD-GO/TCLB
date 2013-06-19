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
                