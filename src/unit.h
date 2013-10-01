#ifndef UNIT_H
#define UNIT_H

#include <string>
#include <map>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
// m s kg K



std::string strFromDouble(double val);

const int m_unit = 7;
const std::string m_units[]={"m","s","kg","K", "x", "y", "z"};

class UnitVal {
public:
  double val;
  int uni[m_unit];
  inline UnitVal(double v) {
      val=v;
      for (int i=0; i<m_unit; i++) uni[i]=0;
  };
  inline void base_unit(int k) {
    if ((k>=0) && (k<m_unit)) {
      val=1;
      for (int i=0; i<m_unit; i++) uni[i]=0;
      uni[k]=1;
    } else {
      printf("Wrong number of unit at initializer\n");
      throw(std::string("Wrong number of unit at initializer"));
    }
  };
  inline UnitVal() {
    val=0;
    for (int i=0; i<m_unit; i++) uni[i]=0;
  };
  inline UnitVal(const UnitVal& e) {
    val=e.val;
    for (int i=0; i<m_unit; i++) uni[i]=e.uni[i];
  };
  inline UnitVal operator* (const UnitVal& A) const {
      UnitVal B;
      B.val = val * A.val;
      for (int i=0; i<m_unit; i++) B.uni[i]=uni[i] + A.uni[i];
      return B;
  };
  inline UnitVal pow (int n) {
      UnitVal B;
      B.val = std::pow(val,n);
      for (int i=0; i<m_unit; i++) B.uni[i]=uni[i]*n;
      return B;
  };
  inline UnitVal operator/ (const UnitVal& A) const {
      UnitVal B;
      B.val = val / A.val;
      for (int i=0; i<m_unit; i++) B.uni[i]=uni[i] - A.uni[i];
      return B;
  };
  inline UnitVal operator+ (const UnitVal& A) const  {
      UnitVal B;
      B.val = val + A.val;
      for (int i=0; i<m_unit; i++) {
        if (A.uni[i] != uni[i]) {
          printf("Different units in addition\n");
          throw(std::string("Different units in addition"));
        }
        B.uni[i] = A.uni[i];
      }
      return B;
  };
  inline void operator= (double v) {
      val = v;
      for (int i=0; i<m_unit; i++) uni[i]=0;
  };
  inline void operator= (const UnitVal & A) {
      val = A.val;
      for (int i=0; i<m_unit; i++) uni[i]=A.uni[i];
  };
  inline bool sameUnit (const UnitVal& A) {
    bool ret = true;
    for (int i=0; i<m_unit; i++) ret = ret && (uni[i] == A.uni[i]);
    return ret;
  };
  inline void print () const{
    printf("%lg [ ",val);
    for (int i=0; i<m_unit; i++) printf("%s^%d ",m_units[i].c_str(),uni[i]);
    printf("]\n");
  };
  inline std::string toString () const{
    std::string str = strFromDouble(val);
    std::string div = "/";
    for (int i=0; i<m_unit; i++) if (uni[i] > 0) {
      str = str + m_units[i] + strFromDouble(uni[i]);
    }
    for (int i=0; i<m_unit; i++) if (uni[i] < 0) {
      str = str + div + m_units[i] + strFromDouble(-uni[i]);
      div = "";
    }
    return str;
  };
};

inline UnitVal pow( UnitVal & A, int b) {
  return A.pow(b);  
}

class UnitVar;

class UnitEnv {
  std::map < std::string, UnitVal > units;
  std::map < std::string, UnitVal > gauge;
  double scale[m_unit];
public:
  UnitEnv ();
  UnitVal readUnitOne( std::string val );
  UnitVal readUnitAlpha( std::string val , int p);
  UnitVal readUnit( std::string val );
  UnitVal readText( std::string val );
  inline UnitVal operator() (std::string str) { return readText(str); };
  inline double si(const UnitVal & v) {return v.val;};
  inline double alt(const UnitVal & v) {
    double ret = v.val;
    for (int i=0; i<m_unit; i++) ret *= pow(scale[i],v.uni[i]);
    return ret;
  };
  inline double si(const std::string str) {return readText(str).val;};
  inline double alt(const std::string str) {
    double ret = 0;
    int i=0, j=0;
    while (str[i]) {
      switch(str[j]) {
      case '-':      
      case '+':
      case '\0':
        if (j>i) {
          UnitVal v = readText(str.substr(i,j-i));
          ret += alt(v);
        }
        i = j;
        break;
      }
      j++;
    }                                                                                                               
    return ret;
  };
  inline double si(const std::string str, double def) { if (str.length() > 0) return si(str); else return def;};
  inline double alt(const std::string str, double def) { if (str.length() > 0) return alt(str); else return def;};
  void setUnit(std::string name, const UnitVal & v, double v2);
  void setUnit(std::string name, const UnitVal & v);
  void makeGauge();
  void printGauge();
};

class UnitVar : public UnitVal {
  UnitEnv * env;
public:
  inline UnitVar(UnitEnv & env_): env(&env_) {};
  inline void operator = (const UnitVal & A) {
      val = A.val;
      for (int i=0; i<m_unit; i++) uni[i]=A.uni[i];
  };
  inline void operator = (std::string str) {
    *this = env->readText(str);
  };
  inline double si () {
    return env->si(*this);
  };
  inline double alt () {
    return env->alt(*this);
  };
};

#endif