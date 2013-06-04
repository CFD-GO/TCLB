#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

class DataLine{
    
    private:
    Node* nodes;
    int size;
    vector<int> cord_x;
    vector<int> cord_y;
    
    public:
    DataLine();
    ~DataLine();
    void readCords(istream &o);
    void readCords(istream &o, int,int);
    void dropRho(ostream &o, Lattice *lattice);
    void showCords();
    //void dropPresInd((ifstream &o);
};

void DataLine::readCords(istream &o, int dx, int dy){
    
    int x, y;
    while (! o.eof()) {
        o >> x >> y;
        cord_x.push_back(x+dx);
        cord_y.push_back(y+dy);
    }
    size = cord_x.size();
    nodes = new Node[size];
}

void DataLine::readCords(istream &o){
    readCords(o,0,0);
}

void DataLine::showCords(){
        cout << "cords"<<endl;
    for ( int i=0; i<size; ++i){
        cout << cord_x[i] << " "<< cord_y[i] << endl;}
    }
    
DataLine::~DataLine(){
    if (size != 0) delete[] nodes;
}

DataLine::DataLine(){
    nodes = NULL;
    size = 0;
}
    
void DataLine::dropRho(ostream &o, Lattice *lattice){
    if (size == 0) return;
    for (int i = 0 ; i< size; ++i){
//        lattice->GetRegion(cord_x[i],cord_y[i], 1, 1, &nodes[i]);
//        o << setprecision(8) << nodes[i].getRho() << " ";
    }
    o << std::endl;
}
    

    