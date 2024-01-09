#include "pugixml.hpp"
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <memory>
#include <math.h>

std::string getPath (const std::string& str)
{
  size_t found;
  found = str.find_last_of("/");
  if (found != std::string::npos) return str.substr(0,found+1);
  return "";
}

struct base64decoder {
	static const char *base64char;
	int rev64[256];
	base64decoder() {
		for (int i = 0; i < 256; i++)
			rev64[i] = 666;
		for (int i = 0; i < 64; i++)
			rev64[((unsigned char *)base64char)[i]] = i;
		rev64[(unsigned char)'='] = 0;
	};
	void dc64(const unsigned char *txt, unsigned char *optr, int n) {
		int v;
		while (n > 0) {
			v = rev64[txt[0]];
			v <<= 6;
			v += rev64[txt[1]];
			v <<= 6;
			v += rev64[txt[2]];
			v <<= 6;
			v += rev64[txt[3]];

			if (n > 2) optr[2] = v & 0xFF;
			v >>= 8;
			if (n > 1) optr[1] = v & 0xFF;
			v >>= 8;
			if (n > 0) optr[0] = v;
			n -= 3;
			optr += 3;
			txt += 4;
		}
	}

	void decode64(const char *txt, void **optr, int len) {
		int nlen;
		unsigned char *ptr;
		txt += 1;
		dc64((unsigned char *)txt, (unsigned char *)&nlen, 4);
		txt += 8;
		assert(len == nlen);
		ptr = (unsigned char *)malloc(nlen);
		dc64((unsigned char *)txt, ptr, nlen);
		*optr = ptr;
	}
};

const char *base64decoder::base64char = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

struct TabBase {
	static base64decoder b64;
	int dx, dy, dz, nx, ny, nz, comp;
	std::string fname;
	std::string ftype;
	size_t size;
	size_t totsize;
	TabBase(int dx_, int dy_, int dz_, int nx_, int ny_, int nz_, int comp_, std::string fname_, std::string ftype_) :
		dx(dx_), dy(dy_), dz(dz_), nx(nx_), ny(ny_), nz(nz_), comp(comp_), fname(fname_), ftype(ftype_) {
		size = 1L * (nx - dx) * (ny - dy) * (nz - dz);
		totsize = size * comp;
	};
	size_t idx(const int& x, const int& y, const int& z, const int& c) const {
		return c + comp*(x - dx + nx * (y - dy + ny * (z - dz + 0L)));
	}
	virtual double compare(TabBase * other, const int& delta_x=0, const int& delta_y=0, const int& delta_z=0) = 0;
	virtual void read_piece(int pdx, int pdy, int pdz, int pnx, int pny, int pnz, pugi::xml_node node) = 0;
};

base64decoder TabBase::b64;

int safe_mod(const int& a, const int& b) {
	int ret = a % b;
	if (ret < 0) return b+ret;
	return ret;
}

template <typename T>
struct Tab : public TabBase {
	std::vector<T> tab;
	double compare(TabBase * other_, const int& delta_x, const int& delta_y, const int& delta_z) {
		assert(ftype == other_->ftype);
		Tab<T>* other = static_cast< Tab<T>* > (other_);
		assert(nx - dx == other->nx - other->dx);
		assert(ny - dy == other->ny - other->dy);
		assert(nz - dz == other->nz - other->dz);
		assert(comp == other->comp);
		int dlx = safe_mod(delta_x, nx);
		int dly = safe_mod(delta_y, ny);
		int dlz = safe_mod(delta_z, nz);
		double diff=0.0;
		for (int z = dz; z < nz; z++) {
			for (int y = dy; y < ny; y++) {
				for (int x = dx; x < nx; x++) {
					for (int c = 0; c < comp; c++) {
						int x_ = ((x-dx+dlx) % (nx-dx)) + other->dx;
						int y_ = ((y-dy+dly) % (ny-dy)) + other->dy;
						int z_ = ((z-dz+dlz) % (nz-dz)) + other->dz;
						double vdiff = tab[idx(x,y,z,c)] - other->tab[other->idx(x_,y_,z_,c)];
						vdiff = fabs(vdiff);
						if (vdiff > diff) diff = vdiff;
					}
				}
			}
		}	
		return diff;
	}
	Tab(int dx_, int dy_, int dz_, int nx_, int ny_, int nz_, int comp_, std::string fname_, std::string ftype_) :
		TabBase(dx_, dy_, dz_, nx_, ny_, nz_, comp_, fname_, ftype_) {
		tab.resize(totsize);
	}
	void read_piece(int pdx, int pdy, int pdz, int pnx, int pny, int pnz, pugi::xml_node node) {
		assert(fname == node.attribute("Name").value());
		assert(ftype == node.attribute("type").value());
		assert(std::string("binary") == node.attribute("format").value());
		assert(std::string("base64") == node.attribute("encoding").value());
		assert(pdx >= dx);
		assert(pdy >= dy);
		assert(pdz >= dz);
		assert(pnx <= nx);
		assert(pny <= ny);
		assert(pnz <= nz);
		size_t psize = 1L * (pnx - pdx) * (pny - pdy) * (pnz - pdz) * comp;
		T *ptr;
		b64.decode64(node.child_value(), (void **)&ptr, psize * sizeof(T));
		T* tmp = ptr;
		for (int z = pdz; z < pnz; z++) {
			for (int y = pdy; y < pny; y++) {
				for (int x = pdx; x < pnx; x++) {
					for (int c = 0; c < comp; c++) {
						T v = tmp[0];
						tab[idx(x,y,z,c)] = v;
						tmp++;
					}
				}
			}
		}
		free(ptr);		
	}
};

struct Tabs {
	std::string filename;
	std::string path;
	int dx, dy, dz, nx, ny, nz;
	typedef TabBase* TabBasePtr;
	typedef std::map<std::string, TabBasePtr> TabMap;
	TabMap tab;
	
	Tabs(std::string filename_) : filename(filename_) {
		path = getPath(filename);
		printf("Reading %s\n", filename.c_str());
		pugi::xml_document file;
		pugi::xml_node el;
		file.load_file(filename.c_str());
		el = file.child("VTKFile");
		assert(el);
		el = el.child("PImageData");
		assert(el);
		{
			const char *reg;
			reg = el.attribute("WholeExtent").value();
			sscanf(reg, "%d %d %d %d %d %d", &dx, &nx, &dy, &ny, &dz, &nz);
		}
		printf("    Fields: ");
		pugi::xml_node tcd = el.child("PCellData");
		for (pugi::xml_node it = tcd.child("PDataArray"); it; it = it.next_sibling("PDataArray")) {
			std::string ftype = it.attribute("type").value();
			std::string fname = it.attribute("Name").value();
			int fcomp = it.attribute("NumberOfComponents").as_int();
			if (ftype == "Float64") {
				tab[fname] = new Tab<double>(dx,dy,dz,nx,ny,nz,fcomp,fname,ftype);
			} else if (ftype == "Float32") {
				tab[fname] = new Tab<float>(dx,dy,dz,nx,ny,nz,fcomp,fname,ftype);
			} else if (ftype == "UInt32") {
				tab[fname] = new Tab<unsigned int>(dx,dy,dz,nx,ny,nz,fcomp,fname,ftype);
			} else if (ftype == "UInt16") {
				tab[fname] = new Tab<unsigned short int>(dx,dy,dz,nx,ny,nz,fcomp,fname,ftype);
			} else if (ftype == "UInt8") {
				tab[fname] = new Tab<unsigned char>(dx,dy,dz,nx,ny,nz,fcomp,fname,ftype);
			} else {
				printf("Unknown field type: %s\n", ftype.c_str());
				exit(-1);
			}
			printf("%s, ", fname.c_str());
		}
		printf("\n");
		for (pugi::xml_node it = el.child("Piece"); it; it = it.next_sibling("Piece"))
		{
			read_piece(it);
		}
	}

	void read_piece(pugi::xml_node el) {
		size_t psize = 0;
		int pdx, pdy, pdz, pnx, pny, pnz;
		{
			const char *reg;
			reg = el.attribute("Extent").value();
			sscanf(reg, "%d %d %d %d %d %d", &pdx, &pnx, &pdy, &pny, &pdz, &pnz);
			psize = (pnx - pdx) * (pny - pdy) * (pnz - pdz);
		}
		std::string filename = path + el.attribute("Source").value();
		printf("    Piece: %s (%ld)\n", filename.c_str(), psize);
		pugi::xml_document pfile;
		pugi::xml_node pel;
		pfile.load_file(filename.c_str());
		pel = pfile.child("VTKFile");
		assert(pel);
		pel = pel.child("ImageData");
		assert(pel);
		pel = pel.child("Piece");
		assert(pel);
		pugi::xml_node tcd = pel.child("CellData");
		assert(pel);
		printf("        Fields: ");
		for (pugi::xml_node it = tcd.child("DataArray"); it; it = it.next_sibling("DataArray")) {
			std::string fname = it.attribute("Name").value();
			if (tab.find(fname) != tab.end()) {
				tab[fname]->read_piece(pdx, pdy, pdz, pnx, pny, pnz, it);
			} else {
				printf("No: %s\n", fname.c_str());
			}
			printf("%s, ", fname.c_str());
		}
		printf("\n");
	}

};

int main(int argc, char *argv[]) {
	double eps = 1e-6;
	int delta_x=0, delta_y=0, delta_z=0;
	switch (argc) {
		case 7: sscanf(argv[6], "%d", &delta_z);
		case 6: sscanf(argv[5], "%d", &delta_y);
		case 5: sscanf(argv[4], "%d", &delta_x);
		case 4: sscanf(argv[3], "%lf", &eps);
		case 3: break;
		default:
			printf("usage: compare file1.pvti file2.pvti [epsilon] [delta_x delta_y delta_z]\n");
			return -1;
	}
	Tabs tabs1(argv[1]);
	Tabs tabs2(argv[2]);
	printf("Epsilon: %lg, Delta: %d, %d, %d\n", eps, delta_x, delta_y, delta_z);
	
	std::set< std::string > names;
	for (Tabs::TabMap::iterator it = tabs1.tab.begin(); it != tabs1.tab.end(); it++) names.insert(it->first);
	for (Tabs::TabMap::iterator it = tabs2.tab.begin(); it != tabs2.tab.end(); it++) names.insert(it->first);
	bool result = true;
	for (std::set< std::string >::iterator it = names.begin(); it != names.end(); it++) {
		std::string name = *it;
		if (tabs1.tab.find(name) == tabs1.tab.end()) {
			printf("%s not in first file\n", name.c_str());
			result = false;
		} else if (tabs2.tab.find(name) == tabs2.tab.end()) {
			printf("%s not in second file\n", name.c_str());
			result = false;
		} else {
			double diff = tabs1.tab[name]->compare(tabs2.tab[name],delta_x,delta_y,delta_z);
			printf("%s: Max difference: %lg", name.c_str(), diff);
			double auto_eps;
			if (tabs1.tab[name]->ftype == "Float64") {
				auto_eps = 2.22e-16;
			} else if (tabs1.tab[name]->ftype == "Float32") {
				auto_eps = 1.19e-07;
			} else {
				auto_eps = 0;
			}
			if (auto_eps != 0) {
				printf(" = %.1lf * %lg", diff / auto_eps, auto_eps);
			}
			if (diff > auto_eps * eps) {
				printf(" --- WRONG\n");
				result = false;
			} else {
				printf(" --- OK\n");
			}		
		}
	}
	if (result) {
		return 0;
	} else {
		return -1;
	}
}
