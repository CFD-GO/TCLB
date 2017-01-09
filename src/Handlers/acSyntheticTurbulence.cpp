#include "acSyntheticTurbulence.h"
std::string acSyntheticTurbulence::xmlname = "SyntheticTurbulence";
#include "../HandlerFactory.h"

int acSyntheticTurbulence::ReadWaveNumer (std::string name, double * var) {
		int set = 0;
		double val = *var;
		pugi::xml_attribute attr;
		attr = node.attribute((name + "WaveLength").c_str());
		if (attr) {
			set++;
			val = 1.0 / solver->units.alt(attr.value());
		}
		attr = node.attribute((name + "WaveNumber").c_str());
		if (attr) {
			set++;
			val = solver->units.alt(attr.value());
		}
		attr = node.attribute((name + "WaveFrequency").c_str());
		if (attr) {
			set++;
			val = solver->units.alt(attr.value())*8*atan(1.0);
		}
		if (set) {
			if (set > 1) {
				NOTICE("Only one of WaveLength, WaveNumber or Frequency, can be set for a Turbulence parameter \"%s\". Taking the last.\n", name.c_str());
			}
			*var = val;
			return 0;
		} else {
			return -1;
		}
	}


int acSyntheticTurbulence::Init () {
		int nmodes;
		Action::Init();
		pugi::xml_attribute attr = node.attribute("Modes");
		if (attr) {
			nmodes = attr.as_int();
		} else {
		        nmodes = 100;
		}
		solver->lattice->ST.resize(nmodes);

		std::string spread;
		attr = node.attribute("Spread");
		if (! attr) {
			spread = "Even";
		} else {
			spread = attr.value();
		}
		if (spread == "Even") {
			solver->lattice->ST.setSpread(EvenSpread);
		} else if (spread == "Log") {
			solver->lattice->ST.setSpread(LogSpread);
		} else if (spread == "Quantile") {
			solver->lattice->ST.setSpread(QuantileSpread);
		} else {
			ERROR("Unknown spread type \"%s\" in %s\n", spread.c_str(), node.name());
			ERROR("Avaliable: Even, Log and Quantile\n");
			return -1;
		}
	
		std::string spec;
		attr = node.attribute("Spectrum");
		if (! attr) {
			spec = "Von Karman";
		} else {
			spec = attr.value();
		}
		if (spec == "Von Karman") {
			double mainWN, diffWN, maxWN, minWN;
			if (ReadWaveNumer("Main", &mainWN)) {
				ERROR("Must provide MainWaveNumber for synthetic turbulence Von Karman spectrum\n");		
				return -1;
			}
			if (ReadWaveNumer("Diffusion", &diffWN)) {
				ERROR("Must provide DiffusionWaveNumber for synthetic turbulence Von Karman spectrum\n");
				return -1;
			}
			maxWN = 8*atan(1)/4; // 2*pi on 4 elements
			ReadWaveNumer("Shortest", &maxWN);
			minWN = mainWN / 2; // Default longest wave is twice the size of the main
			ReadWaveNumer("Longest", &minWN);
			output("Setting Von Karman spectrum with main length=%lg and diffusion length=%lg. With spread of %d modes across lengths %lg - %lg\n", 1/mainWN, 1/diffWN, nmodes, 1/maxWN, 1/minWN); 
			solver->lattice->ST.setVonKarman(mainWN, diffWN, minWN, maxWN);
		} else if (spec == "One Wave") {
			double WN;
			nmodes = 1;
			solver->lattice->ST.resize(nmodes);
			if (ReadWaveNumer("", &WN)) {
				ERROR("Must provide WaveNumber for synthetic turbulence\n");
				return -1;
			}
			output("Setting one-wave spectrum with length=%lg\n", 1/WN); 
			solver->lattice->ST.setOneWave(WN);
		} else {
			ERROR("Unknown spectrum type \"%s\" in %s\n", spec.c_str(), node.name());
			return -1;
		}
		{
			double timeWN;			
			if (ReadWaveNumer("Time", &timeWN)) {
                                ERROR("Must provide TimeWaveNumber for synthetic turbulence\n");
                                return -1;
                        }
			solver->lattice->ST.setTimeScale(timeWN);
		}
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acSyntheticTurbulence > >;
