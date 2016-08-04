#include "Consts.h"
#include "Global.h"
#include <vector>
#include "ZoneSettings.h"

int ZoneSettings::dumpToFile(const char* filename) {
  output("Writing Zone Settings: %s\n",filename);
  FILE* f = fopen(filename,"w");
  fprintf(f,"Iteration");
  for (int z=0; z<MaxZones; z++) {
    for (int s=0; s<ZONESETTINGS; s++) {
      fprintf(f,", S.%d.%d",s,z);
      fprintf(f,", D.%d.%d",s,z);
    }
  }
  fprintf(f,"\n");
  for (size_t it=0; it<len; it++) {
    fprintf(f,"%ld",it);
    for (int z=0; z<MaxZones; z++) {
      for (int s=0; s<ZONESETTINGS; s++) {
        for (int d=0; d<2; d++) {
          int i = s+ZONESETTINGS*z + DT_OFFSET*d;
          double val;
          if (cpuValues[i] == NULL) {
            val = cpuConst[i];
          } else {
            val = cpuValues[i][it];
          }
          fprintf(f,", %lg",val);
        }
      }
    }
    fprintf(f,"\n");
  }
  fclose(f);
  return 0;
}
