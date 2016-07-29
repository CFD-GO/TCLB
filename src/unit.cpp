#include <string>
#include <map>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <sstream>
#include "unit.h"
#define IND(x,y) ((y)*n+(x))

void GaussSolve (double A[], double b[], double x[], int n)
{
  int i, j, k;

  for (k=0; k<n-1; k++)
  {
    if (A[IND(k,k)] == 0.0)
      for (i=k+1; i<n; i++)
        if (A[IND(i,k)] != 0.0)
        {
          double tmp;
          for (j=0; j<n; j++)
          {
            tmp = A[IND(k,j)];
            A[IND(k,j)] = A[IND(i,j)];
            A[IND(i,j)] = tmp;
          }
          tmp = b[k];
          b[k] = b[i];
          b[i] = tmp;
          break;
        }

    for (i=k+1; i<n; i++)
    {
      double m = A[IND(i,k)] / A[IND(k,k)];
      for (j=0; j<n; j++)
      {
        A[IND(i,j)] -= m * A[IND(k,j)];
      }
      b[i] -= m * b[k];
    }
  }

  for (i=n-1; i>=0; i--)
  {
    double sum = 0.0;
    for (j=n-1; j>=i+1; j--)
      sum += A[IND(i,j)]*x[j];
    x[i] = (b[i] - sum)/A[IND(i,j)];
  }
}
// m s kg K

std::string strFromDouble(double val) {
    std::ostringstream strs;
    strs << val;
    return strs.str();
}

  UnitEnv::UnitEnv (){
    for (int i=0; i<m_unit; i++) scale[i]=1;
    UnitVal A(1);
    for (int i=0; i<m_unit; i++ ) {
      A.base_unit(i);
      units[m_units[i]]=A;
    }

    //special units, derivatives
    units["N"]=readText("1kgm/s2");
    units["Pa"]=readText("1N/m2");
    units["J"]=readText("1Nm");
    units["W"]=readText("1J/s");
    units["V"]=readText("1kgm2/t3/A");
    units["C"]=readText("1tA");

    //prefixes
    
    units["nm"]=readText("1e-9m"); 
    units["um"]=readText("1e-6m");
    units["mm"]=readText("1e-3m");
    units["cm"]=readText("1e-2m");
    units["km"]=readText("1e+3m");

    units["h"]=readText("3600s");
    units["ns"]=readText("1e-9s");
    units["us"]=readText("1e-6s");
    units["ms"]=readText("1e-3s");

    units["g"]=readText("1e-3kg");
    units["mg"]=readText("1e-6kg");
    


    units["d"]=atan(1.0)*4.0/180.;
    units["%"]=1./100.;
    units["An"] = 6.022*100000000000000000000000.;
  };
  UnitVal UnitEnv::readUnitOne( std::string val ) {
    if (units.count(val) > 0) {
      return units[val];
    } else {
      return 0;
    }
  };
  UnitVal UnitEnv::readUnitAlpha( std::string val , int p) {
    UnitVal ret,ret1,ret2;
    int i=0, j=1;
    ret1 = readUnitOne(val.substr(0,1));
    if (val.length() < 2) {
      ret = pow(ret1,p);
    } else {
      ret1 = ret1 * readUnitAlpha(val.substr(1),p);
      ret2 = readUnitOne(val.substr(0,2));
      if (val.length() > 2) {
        ret2 = ret2 * readUnitAlpha(val.substr(2),p);
      } else {
        ret2 = pow(ret2,p);
      }
      if (ret1.val == 0) {
        if (ret2.val == 0) {
          ret = 0;
        } else {
          ret = ret2;
        }
      } else {
        if (ret2.val == 0) {
          ret = ret1;
        } else {
          if (val[0] == 'm') {
            warning("Disambiguous unit: \"%s\". Interpreting \"m\" as \"mili\"\n", val.c_str());
            ret = ret2;
          } else {
            error("Disambiguous unit: \"%s\"\n", val.c_str());
            throw(std::string("Disambiguous"));
            ret = 0;
          }
        }
      }
    }
    return ret;
  };
  UnitVal UnitEnv::readUnit( std::string val ) {
    UnitVal last, ret;
    int i,j,k,l,p,w=1;
    ret = 1.;
    i=0;
    while (val[i]) {
      j=i;
      while (isalpha(val[i])) i++;
      k=i;
      while (isdigit(val[i])) i++;
      l=i;
      p=1;
      if (l-k > 0) {
        p = atoi(val.substr(k,l-k).c_str());
      }
      if (k-j > 0) {
        last = readUnitAlpha(val.substr(j,k-j).c_str(), p);
      } else {
        last = 1.0;
      }
      if (w > 0) {
        ret = ret * last;
      } else {
        ret = ret / last;
      }
      j=i;
      while (!isalnum(val[i]) && (val[i] > 0)) i++;
      if (i-j > 1) {
        error("Too many non-alpha-numeric characters in units: \"%s\"\n", val.substr(j,i-j).c_str());
        throw(std::string("Wrong non-alpha-numeric in unit"));
      }
      if (i-j == 1) {
        if (val[j] == '/') {
          w = -1;
        } else {
          error("Only \"/\" allowed in units: \"%c\"\n", val[j]);
          throw(std::string("Wrong non-alpha-numeric in unit"));
        }
      }
    }
    return ret;
  };
  UnitVal UnitEnv::readText( std::string val ) {
    UnitVal ret = 1.;
    std::string num = "", unit = "";
    int i=0, j=1;
    while (j) switch(val[i]) {
      case '-':  
      case '+':  
      case '0':  
      case '1':  
      case '2':  
      case '3':  
      case '4':  
      case '5':  
      case '6':  
      case '7':  
      case '8':  
      case '9':  
      case '.':  
      case 'e':  
      case 'E':
        i++;
        break;
      default:
        j=0;
    }
    unit = val.substr(i);
    ret = readUnit(unit);
    if (i > 0) {
      num=val.substr(0, i);
      ret = ret * ((UnitVal) atof(num.c_str()));
    }
    return ret;
  };
  void UnitEnv::setUnit(std::string name, const UnitVal & v, double v2) {
    gauge[name] = v/UnitVal(v2);
  }
  void UnitEnv::setUnit(std::string name, const UnitVal & v) {
    gauge[name] = v;
  }
  void UnitEnv::makeGauge() {
    double Mat[m_unit*m_unit];
    double x[m_unit], b[m_unit];
    int i=0,j;
    for(i=0;i<m_unit*m_unit;i++) {
      Mat[i]=0;
    }
    i=0;
    for(std::map<std::string, UnitVal>::iterator el=gauge.begin();el!=gauge.end();el++) {
      UnitVal v = el->second;
      for (j=0;j<m_unit;j++) {
        Mat[m_unit*j+i] = v.uni[j];
      }
      b[i] = log(v.val);
      i++;
    }
    for (int j=0;j<m_unit;j++) {
      int v=0;
      for (int l=0;l<i;l++) {
        if (Mat[m_unit*j+l] != 0) v=1;
      }
      if (!v) {
        if (i >= m_unit) {
          ERROR("Gauge variables over-constructed\n");
          throw(std::string("Wrong number of gauge variables"));
        }
        Mat[m_unit*j+i]=1;
        b[i]=0;
        i++;
      }
    }
    if (i < m_unit) {
      ERROR("Gauge variables under-constructed\n");
      throw(std::string("Wrong number of gauge variables"));
    }
    GaussSolve(Mat,b,x,m_unit);
    for (int j=0;j<m_unit;j++) {
      scale[j] = exp(-x[j]);
    }
  }
  void UnitEnv::printGauge() {
    output("/---------------[ GAUGE ]-----------------\n");
    for(std::map<std::string, UnitVal>::iterator el=gauge.begin();el!=gauge.end();el++) {
      UnitVal v = el->second;
      output("|  %s\n", v.tmp_str());
    }
    output("------------------------------------------\n");
    for (int j=0;j<m_unit;j++) {
      output("| 1 %s = %lf units\n", m_units[j].c_str(), scale[j]);
    }
    output("\\-----------------------------------------\n");
  }

