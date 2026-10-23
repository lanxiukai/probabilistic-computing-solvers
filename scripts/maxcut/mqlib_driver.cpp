// MQLib's public callback API supplies bounded search and exact-work replay.
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "heuristics/maxcut/burer2002.h"
#include "heuristics/maxcut/festa2002.h"
#include "problem/max_cut_heuristic.h"

class Recorder final : public MaxCutCallback {
public:
    double seconds;
    long long fixed_calls, quality_calls, calls = 0, last_eligible = 0, first_stop = 0;
    double best_value = 0;
    std::chrono::steady_clock::time_point started;
    std::vector<int> best;
    Recorder(int size, double limit, long long fixed, long long quality)
        : seconds(limit), fixed_calls(fixed), quality_calls(quality), best(size, -1) {}

    bool record(const MaxCutSimpleSolution& solution, bool new_best, double runtime) {
        runtime = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        ++calls;
        bool eligible = quality_calls >= 0 ? calls <= quality_calls : runtime <= seconds;
        if (eligible) last_eligible = calls;
        if (eligible && solution.get_weight() > best_value) {
            best_value = solution.get_weight();
            best = solution.get_assignments();
        }
        if (new_best) {
            std::cout << "o " << calls << " " << std::llround(solution.get_weight())
                      << " " << eligible << " " << runtime << std::endl;
        }
        bool proceed = fixed_calls >= 0 ? calls < fixed_calls : runtime < seconds;
        if (!proceed && first_stop == 0) first_stop = calls;
        return proceed;
    }
    bool Report(const MaxCutSimpleSolution& s, bool improved, double runtime) override {
        return record(s, improved, runtime);
    }
    bool Report(const MaxCutSimpleSolution& s, bool improved, double runtime, int) override {
        return record(s, improved, runtime);
    }
};

void print_bits(const char* label, const std::vector<int>& assignments) {
    std::cout << label << " ";
    for (int spin : assignments) std::cout << (spin == 1 ? 1 : 0);
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 7) return 2;
    std::string method(argv[1]);
    if (method != "BURER2002" && method != "FESTA2002VNS") return 2;
    MaxCutInstance instance(argv[2]);
    unsigned seed = std::stoul(argv[3]);
    double seconds = std::stod(argv[4]);
    long long fixed = std::stoll(argv[5]), quality = std::stoll(argv[6]);
    if (!(seconds > 0) || fixed == 0 || fixed < -1 || quality < -1) return 2;
    std::srand(seed);
    Recorder callback(instance.get_size(), seconds, fixed, quality);
    std::cout << std::setprecision(17) << "c benchmark-search-ready" << std::endl;
    char resume;
    if (!(std::cin >> resume) || resume != 'g') return 2;
    callback.started = std::chrono::steady_clock::now();
    std::unique_ptr<MaxCutHeuristic> algorithm;
    if (method == "BURER2002") {
        algorithm.reset(new Burer2002(instance, seconds, true, &callback));
    } else {
        algorithm.reset(new Festa2002VNS(instance, seconds, true, &callback));
    }
    if (!algorithm->IsHistoryValid()) return 3;
    std::cout << "c benchmark-calls " << callback.calls << std::endl;
    std::cout << "c first-stop-call " << callback.first_stop << std::endl;
    std::cout << "c quality-calls " << callback.last_eligible << std::endl;
    std::cout << "c best-cut " << std::llround(callback.best_value) << std::endl;
    std::cout << "c final-cut " << std::llround(algorithm->get_best()) << std::endl;
    print_bits("v", callback.best);
    print_bits("f", algorithm->get_best_solution().get_assignments());
    return 0;
}
