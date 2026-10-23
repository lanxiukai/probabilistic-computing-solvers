// Private benchmark adapter for the unmodified MIT-licensed ldpc C++ headers.
// The decoder sees only a parity matrix, error syndrome and nonnegative LLR costs.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "bp.hpp"
#include "osd.hpp"
#include "lsd.hpp"

using Clock = std::chrono::steady_clock;

std::string bit_string(const std::vector<uint8_t>& bits) {
    std::string result;
    for (auto bit : bits) result.push_back(bit ? '1' : '0');
    return result;
}

template <typename T> void read_values(std::ifstream& input, std::vector<T>& values) {
    input.read(reinterpret_cast<char*>(values.data()), sizeof(T) * values.size());
    if (!input) throw std::runtime_error("Truncated decoder input");
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error(
            "Usage: reference input.bin output.jsonl mode iterations alpha schedule");
        const std::string mode = argv[3];
        if (mode != "bp" && mode != "osd" && mode != "lsd")
            throw std::runtime_error("Unknown postprocessor");
        const int iterations = std::stoi(argv[4]);
        const double alpha = std::stod(argv[5]);
        const auto schedule = std::string(argv[6]) == "serial"
            ? ldpc::bp::SERIAL : ldpc::bp::PARALLEL;
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open decoder input");
        char magic[8]{};
        input.read(magic, 8);
        if (std::string(magic, 8) != "LDPCCS01")
            throw std::runtime_error("Invalid decoder input header");
        std::vector<uint32_t> header(4);
        read_values(input, header);
        const int n = header[0], m = header[1], frames = header[2], scale = header[3];
        if (n < 2 || m < 1 || frames < 1 || scale != 256)
            throw std::runtime_error("Invalid decoder dimensions or scale");
        if (mode == "osd" && n > 2048)
            throw std::runtime_error("OSD combination sweep is limited to 2048 variables");
        ldpc::bp::BpSparse matrix(m, n);
        for (int a = 0; a < m; ++a) {
            std::vector<uint32_t> count(1);
            read_values(input, count);
            std::vector<uint32_t> columns(count[0]);
            read_values(input, columns);
            for (auto column : columns) {
                if (column >= static_cast<uint32_t>(n))
                    throw std::runtime_error("Invalid matrix entry");
                matrix.insert_entry(a, column);
            }
        }
        std::ofstream output(argv[2]);
        output << std::setprecision(17);
        for (int frame = 0; frame < frames; ++frame) {
            std::vector<uint32_t> costs(n);
            std::vector<uint8_t> syndrome(m);
            read_values(input, costs);
            read_values(input, syndrome);
            auto started = Clock::now();
            std::vector<double> probabilities(n), score_weights(n), posterior(n);
            for (int i = 0; i < n; ++i) {
                if (costs[i] > 8192) throw std::runtime_error("Unsupported receiver magnitude");
                posterior[i] = static_cast<double>(costs[i]) / scale;
                probabilities[i] = 1.0 / (1.0 + std::exp(posterior[i]));
                // Upstream OSD ranks candidates with -log(p), not log((1-p)/p).
                // Supply exp(-LLR) only to its scoring input, preserving true BP priors.
                score_weights[i] = std::exp(-posterior[i]);
            }
            auto mismatches = [&](std::vector<uint8_t>& bits) {
                auto actual = matrix.mulvec(bits);
                int result = 0;
                for (int a = 0; a < m; ++a) result += actual[a] != syndrome[a];
                return result;
            };
            auto cost = [&](const std::vector<uint8_t>& bits) {
                uint64_t result = 0;
                for (int i = 0; i < n; ++i) result += static_cast<uint64_t>(costs[i]) * bits[i];
                return result;
            };
            std::vector<uint8_t> bp_bits(n, 0), selected(n, 0), post_bits;
            int used_iterations = 0, osd_order = 0;
            bool post_called = false;
            double bp_seconds = 0, post_seconds = 0;
            const bool zero = std::none_of(syndrome.begin(), syndrome.end(),
                                           [](auto x) { return x != 0; });
            if (!zero) {
                ldpc::bp::BpDecoder bp(matrix, probabilities, iterations,
                    ldpc::bp::MINIMUM_SUM, schedule, alpha, 1,
                    ldpc::bp::NULL_INT_VECTOR, 0, false, ldpc::bp::SYNDROME);
                bp_bits = bp.decode(syndrome);
                used_iterations = bp.iterations;
                posterior = bp.log_prob_ratios;
                for (auto value : posterior)
                    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite BP posterior");
                selected = bp_bits;
                bp_seconds = std::chrono::duration<double>(Clock::now() - started).count();
                auto post_start = Clock::now();
                if (mode == "osd") {
                    osd_order = std::min(20, std::max(0, n - m));
                    ldpc::osd::OsdDecoder osd(matrix, ldpc::osd::COMBINATION_SWEEP,
                                              osd_order, score_weights);
                    post_bits = osd.decode(syndrome, posterior);
                    post_called = true;
                } else if (mode == "lsd" && !bp.converge) {
                    ldpc::lsd::LsdDecoder lsd(matrix, ldpc::osd::COMBINATION_SWEEP, 0);
                    post_bits = lsd.lsd_decode(syndrome, posterior, 1, true);
                    post_called = true;
                }
                if (post_called) {
                    if (mismatches(post_bits))
                        throw std::runtime_error("Reference postprocessor returned an invalid syndrome");
                    if (mismatches(selected) || cost(post_bits) < cost(selected))
                        selected = post_bits;
                }
                post_seconds = std::chrono::duration<double>(Clock::now() - post_start).count();
            }
            const auto elapsed = std::chrono::duration<double>(Clock::now() - started).count();
            output << "{\"frame\":" << frame << ",\"iterations\":" << used_iterations
                   << ",\"post_called\":" << (post_called ? "true" : "false")
                   << ",\"osd_order\":" << osd_order
                   << ",\"cost\":" << cost(selected)
                   << ",\"violations\":" << mismatches(selected)
                   << ",\"bp_seconds\":" << bp_seconds
                   << ",\"post_seconds\":" << post_seconds
                   << ",\"decoder_seconds\":" << elapsed
                   << ",\"bp_bits\":\"" << bit_string(bp_bits)
                   << "\",\"selected_bits\":\"" << bit_string(selected)
                   << "\",\"post_bits\":";
            if (post_called) output << "\"" << bit_string(post_bits) << "\"";
            else output << "null";
            output << ",\"posterior\":[";
            for (int i = 0; i < n; ++i) {
                if (i) output << ",";
                output << posterior[i];
            }
            output << "]}\n";
            output.flush();
        }
        char extra;
        if (input.read(&extra, 1)) throw std::runtime_error("Trailing decoder input");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
