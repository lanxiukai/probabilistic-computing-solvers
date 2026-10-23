// Observe native timer checks without changing CFT search decisions.
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

static const void* watched_timer = nullptr;
static std::vector<std::int64_t> clock_values;
static std::vector<std::int64_t> replay_values;
static std::size_t replay_index = 0;
static bool replay_clock = false;

std::int64_t research_clock(const void* object, std::int64_t actual) {
    if (object != watched_timer) return actual;
    if (replay_clock) {
        if (replay_index >= replay_values.size()) throw std::runtime_error("Clock replay exhausted");
        actual = replay_values[replay_index++];
    }
    clock_values.push_back(actual);
    return actual;
}

#include "algorithms/Refinement.hpp"
#include "core/Instance.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("Expected input, output, seed, seconds, trace, replay");
        std::ifstream input(argv[1]);
        int m, n;
        if (!(input >> m >> n) || m <= 0 || m >= 32767 || n <= 0)
            throw std::runtime_error("Invalid covering dimensions");
        cft::Instance instance;
        for (int j = 0; j < n; ++j) {
            int cost, degree;
            if (!(input >> cost >> degree) || cost <= 0 || degree <= 0 || degree > m)
                throw std::runtime_error("Invalid covering column");
            std::vector<cft::ridx_t> rows;
            for (int k = 0; k < degree; ++k) {
                int i;
                if (!(input >> i) || i < 0 || i >= m) throw std::runtime_error("Invalid row index");
                rows.push_back(static_cast<cft::ridx_t>(i));
            }
            instance.costs.push_back(static_cast<cft::real_t>(cost));
            instance.cols.push_back(rows);
        }
        cft::fill_rows_from_cols(instance.cols, static_cast<cft::ridx_t>(m), instance.rows);
        cft::Environment env;
        env.verbose = 0;
        env.seed = std::stoull(argv[3]);
        env.rnd = cft::prng_t(env.seed);
        env.time_limit = std::stod(argv[4]);
        if (env.time_limit <= 0) throw std::runtime_error("Positive time limit required");
        replay_clock = std::string(argv[6]) != "-";
        if (replay_clock) {
            std::ifstream stream(argv[6], std::ios::binary);
            if (!stream) throw std::runtime_error("Clock replay file missing");
            std::int64_t value;
            while (stream.read(reinterpret_cast<char*>(&value), sizeof value)) replay_values.push_back(value);
        }
        watched_timer = &env.timer;
        env.timer.restart();
        const auto start = std::chrono::steady_clock::now();
        auto result = cft::run(env, instance);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        watched_timer = nullptr;
        if (replay_clock && replay_index != replay_values.size())
            throw std::runtime_error("Unused clock replay values");
        std::ofstream clock_file(argv[5], std::ios::binary);
        clock_file.write(reinterpret_cast<const char*>(clock_values.data()), clock_values.size() * sizeof(std::int64_t));
        std::ofstream output(argv[2]);
        output << std::setprecision(17) << "{\"cost\":" << result.sol.cost << ",\"columns\":[";
        for (std::size_t j = 0; j < result.sol.idxs.size(); ++j) {
            if (j) output << ',';
            output << result.sol.idxs[j];
        }
        output << "],\"dual_multipliers\":[";
        for (std::size_t i = 0; i < result.dual.mults.size(); ++i) {
            if (i) output << ',';
            output << result.dual.mults[i];
        }
        output << "],\"native_dual_bound\":" << result.dual.lb
               << ",\"search_seconds\":" << seconds
               << ",\"clock_queries\":" << clock_values.size()
               << ",\"clock_replay\":" << (replay_clock ? "true" : "false") << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
