// Exercise the whole-neighborhood flip with an instrumented partition invariant.
#include <cstdlib>
#include <iostream>
#include <vector>
#include "heuristics/maxcut/festa2002.h"

int main() {
    std::vector<Instance::InstanceTuple> edges;
    edges.push_back(Instance::InstanceTuple(std::pair<int, int>(1, 2), 1));
    edges.push_back(Instance::InstanceTuple(std::pair<int, int>(2, 3), 1));
    edges.push_back(Instance::InstanceTuple(std::pair<int, int>(3, 4), 1));
    MaxCutInstance instance(edges, 4);
    std::srand(1);
    Festa2002Solution base(MaxCutSolution::RandomSolution(instance, nullptr));
    auto result = Festa2002Solution::NeighborhoodSolution(instance, base, 4, nullptr);
    for (int i = 0; i < 4; ++i) {
        if (result.get_assignments()[i] != -base.get_assignments()[i]) return 4;
    }
    std::cout << "partition invariant passed" << std::endl;
    return 0;
}
