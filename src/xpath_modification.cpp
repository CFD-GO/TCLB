#include "Global.h"
#include "xpath_modification.h"

int xpath_modify(pugi::xml_document& doc, pugi::xml_node config, int argc, char * argv[] ) {
    for (int i = 0; i < argc; i++) {
            try {
                    char * add_attribute = NULL;
                    if (strcmp(argv[i], "-s") == 0) {
                        i++;
                        if (i >= argc) {
                                ERROR("no filename for -s\n");
                                return -1;
                        }
                        doc.save_file(argv[i]);
                        continue;
                    } else if (strcmp(argv[i], "-x") == 0) {
                        output("Gracefully exiting\n");
                        return -444;
                    }
                    pugi::xpath_node_set found;
                    if (argv[i][0] == '-' && argv[i][1] == '-') {
                        found = config.select_nodes(".");
                        add_attribute = argv[i] + 2;
                    } else {
                        output("XPATH: %s\n",argv[i]);
                        found = config.select_nodes(argv[i]);
                        output("XPATH: %ld things found\n", found.size());
                        i++;
                        if (i < argc && argv[i][0] == '@') {
                                add_attribute = argv[i] + 1;
                                i++;
                        }
                        if (i >= argc) {
                                ERROR("no operator in xpath evaluation\n");
                                return -1;
                        }
                        if ((add_attribute != NULL) && (strcmp(argv[i], "=") != 0)) {
                                ERROR("Only '=' operator can be used for adding attribute\n");
                                return -1;
                        }
                    }
                    if ((add_attribute != NULL) || (strcmp(argv[i], "=") == 0)) {
                            i++;
                            if (i >= argc) {
                                    ERROR("XPATH: No value supplied to = operator\n");
                                    return -1;
                            } else if (found.size() == 0) {
                                    ERROR("XPATH: Nothing selected for substitution\n");
                                    return -1;
                            } else for (pugi::xpath_node_set::const_iterator it = found.begin(); it != found.end(); ++it) {
                                    bool error = false;
                                    pugi::xml_attribute attr = it->attribute();
                                    if (add_attribute == NULL) {
                                            if (attr) {
                                                    attr.set_value(argv[i]);
                                                    output("XPATH: Set attr %s to \"%s\" in <%s />\n", attr.name(), attr.value(), it->parent().name());
                                            } else error = true;
                                    } else {
                                            pugi::xml_node node = it->node();
                                            if (node) {
                                                    attr = node.attribute(add_attribute);
                                                    if (attr) {
                                                            attr.set_value(argv[i]);
                                                            output("XPATH: Set attr %s to \"%s\" in <%s />\n", attr.name(), attr.value(), node.name());
                                                    } else {
                                                            attr = node.append_attribute(add_attribute);
                                                            attr.set_value(argv[i]);
                                                            output("XPATH: Adding attr %s = \"%s\" to <%s />\n", attr.name(), attr.value(), node.name());
                                                    }
                                            } else error = true;
                                    }
                                    if (error) {
                                            ERROR("XPATH: Operator = can only be used for attributes\n");
                                            ERROR(" eg. Model/Params/@Viscosity = 0 - change attribute\n");
                                            ERROR("     Model/Params @Viscosity = 0 - add or change attribute\n");
                                            return -1;
                                    }
                            }
                    } else if ((strcmp(argv[i], "inject") == 0) || (strcmp(argv[i], "insert") == 0)) {
                            i++;
                            int type = 0;   // 0 - last   - at the end of node
                                            // 1 - first  - at the begining of node
                                            // 2 - after  - after a node
                                            // 3 - before - before a node
                            if (i >= argc) {
                                    ERROR("XPATH: No value or specifier supplied to inject operator\n");
                                    return -1;
                            } else if (strcmp(argv[i], "last") == 0) {
                                    type = 0; i++;
                            } else if (strcmp(argv[i], "first") == 0) {
                                    type = 1; i++;
                            } else if (strcmp(argv[i], "after") == 0) {
                                    type = 2; i++;
                            } else if (strcmp(argv[i], "before") == 0) {
                                    type = 3; i++;
                            }
                            pugi::xml_document doc;
                            pugi::xml_parse_result result;
                            if (i >= argc) {
                                    ERROR("XPATH: No value supplied to inject operator\n");
                                    return -1;
                            }
                            result = doc.load_string(argv[i]);
                            if (!result) {
                                    ERROR("XPATH: Error while parsing inject string: %s\n", result.description());
                                    return -1;
                            }
                            pugi::xml_node node = doc.first_child();
                            if (! node) {
                                    ERROR("XPATH: No XML children in inject string\n");
                                    return -1;
                            } else if (node.next_sibling()) {
                                    ERROR("XPATH: More then one XML child in inject string\n");
                                    return -1;
                            }
                            if (found.size() == 0) {
                                    ERROR("XPATH: Nothing selected for injection\n");
                                    return -1;
                            } else if (found.size() != 1) {
                                    WARNING("XPATH: More then one thing selected for injection\n");
                            }
                            for (pugi::xpath_node_set::const_iterator it = found.begin(); it != found.end(); ++it) {
                                    if (it->node()) {
                                            if (type == 0) {
                                                    it->node().append_copy(node); break;
                                            } else if (type == 1) {
                                                    it->node().prepend_copy(node); break;
                                            } else if (type == 2) {
                                                    it->node().parent().insert_copy_after(node, it->node()); break;
                                            } else if (type == 3) {
                                                    it->node().parent().insert_copy_before(node, it->node()); break;
                                            } else {
                                                    ERROR("XPATH: Unknown type (this should not happen)\n");
                                            }
                                    } else {
                                            ERROR("XPATH: Operator 'insert' can only be used for nodes\n");
                                            return -1;
                                    }
                            }
                    } else if (strcmp(argv[i], "delete") == 0) {
                            for (pugi::xpath_node_set::const_iterator it = found.begin(); it != found.end(); ++it) {
                                    if (it->node()) {
                                            output("XPATH: Removed node: <%s />\n", it->node().name());
                                            it->parent().remove_child(it->node());
                                    } else if (it->attribute()) {
                                            output("XPATH: Removed attribute: %s = %s\n", it->attribute().name(), it->attribute().value());
                                            it->parent().remove_attribute(it->attribute());
                                    }
                            }
                    } else if (strcmp(argv[i], "print") == 0) {
                            for (pugi::xpath_node_set::const_iterator it = found.begin(); it != found.end(); ++it) {
                                    if (it->node()) {
                                            output("XPATH: Node: %s\n", it->node().name());
                                    } else if (it->attribute()) {
                                            output("XPATH: Attr: %s=\"%s\"\n", it->attribute().name(), it->attribute().value());
                                    }
                            }
                    } else {
                            ERROR("Unknown operator in xpath evaluation: %s\n",argv[i]);
                            ERROR("Currently supported: =, print, insert/inject, delete\n");
                            return -1;
                    }
            } catch (pugi::xpath_exception& err) {
                    ERROR("XPATH: parsing error: %s\n", err.what());
                    ERROR("XPATH: Syntax: .../main file.xml XPATH = value (spaces are important)");
                    return -1;
            }
    }
    return 0;
}


