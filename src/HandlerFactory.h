#ifndef HANDLER_FACTORY_H
#define HANDLER_FACTORY_H

#include "Factory.h"
#include "pugixml.hpp"
#include "Handlers/vHandler.h"

typedef Factory< vHandler, pugi::xml_node > HandlerFactory;

#endif