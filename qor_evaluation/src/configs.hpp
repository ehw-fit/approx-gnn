#pragma once

#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <array>
#include "common.hpp"

#include "_components.hpp"
#include "_wirings.hpp"

/// @brief Wiring configuration.
typedef std::function<int(int, int, int, int, int, int, int, int, int, tAdd[])>
    tWiring;

typedef std::tuple<tWiring, tAdd[]> tConfig;

int accurate_add(int a, int b) { return a + b; }

class Configs {
public:
    std::map<std::string, tWiring> wirings;
    std::map<std::string, tAdd> adders;

    Configs() {
        adders["add_ref"] = accurate_add;
    }

    void load();
};