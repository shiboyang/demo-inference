//
// Created by shiby on 23-9-28.
//

#ifndef INFERENCE_DEMO_UTILS_H
#define INFERENCE_DEMO_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>

std::vector<std::string> loadClasses(std::string &classPath) {
    std::vector<std::string> classes;
    std::ifstream infile(classPath);
    assert(infile.is_open() && "Attempting to reading from a file that is not open.");
    std::string name;
    while (!infile.eof()) {
        std::getline(infile, name);
        if (name.length() > 0)
            classes.push_back(name);
    }
    infile.close();

    return classes;
}



#endif //INFERENCE_DEMO_UTILS_H
