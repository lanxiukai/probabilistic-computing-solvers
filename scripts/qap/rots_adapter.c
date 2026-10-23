/* Measurement adapter authored for this project; no upstream move logic here.
 * The separately downloaded/instrumented Taillard algorithm retains its
 * non-commercial-use conditions and acknowledgement requirement.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned long long rots_rng_draws = 0;
static int rots_before_iteration(int iteration);
static void rots_observe(int iteration, int n, int **a, int **b, int *p, int cost,
                         int *best, int best_cost, int **tabu, int **delta);
static void rots_finish(int n, int *p, int cost, int *best, int best_cost, int **tabu);
static int rots_quiet_printf(const char *format, ...) { (void)format; return 0; }
#define printf rots_quiet_printf
#define main unused_upstream_main
#include "instrumented.c"
#undef main
#undef printf

static volatile sig_atomic_t requested_signal = 0;
static double search_start, search_limit, search_seconds, checkpoint_time;
static int completed_iterations = 0, observation_count = 0, previous_best = INT_MAX;
static int verify_caches = 0, final_cost = 0, final_best_cost = 0;
static int *final_permutation = NULL, *final_best = NULL;
static FILE *observations = NULL;
static const char *evidence_directory = NULL;

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value)) exit(70);
    return (double)value.tv_sec + 1e-9 * (double)value.tv_nsec;
}

static void interrupted(int signum) { requested_signal = signum; }

static void vector_json(FILE *stream, const int *values, int count) {
    fputc('[', stream);
    for (int i = 0; i < count; ++i) fprintf(stream, "%s%d", i ? "," : "", values[i]);
    fputc(']', stream);
}

static void rng_get(int values[6]) {
    values[0] = x10; values[1] = x11; values[2] = x12;
    values[3] = x20; values[4] = x21; values[5] = x22;
}

static void rng_set(const int values[6]) {
    x10 = values[0]; x11 = values[1]; x12 = values[2];
    x20 = values[3]; x21 = values[4]; x22 = values[5];
}

static int64_t original_cost(int n, int **a, int **b, const int *p) {
    int64_t value = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        value += (int64_t)a[i][j] * b[p[i]][p[j]];
    return value;
}

static void checkpoint(const char *kind, int n, const int *p, int cost,
                       const int *best, int best_cost, int **tabu, double elapsed) {
    char filename[4096];
    int length = snprintf(filename, sizeof(filename), "%s/checkpoint-%s-%d.json",
                          evidence_directory, kind, completed_iterations);
    if (length < 0 || (size_t)length >= sizeof(filename)) exit(71);
    FILE *stream = fopen(filename, "w");
    if (!stream) exit(72);
    int rng[6]; rng_get(rng);
    fprintf(stream, "{\"iteration\":%d,\"seconds\":%.17g,\"cost\":%d,\"best_cost\":%d,"
            "\"observation_count\":%d,\"rng_draws\":%llu,\"signal\":%d,\"rng\":",
            completed_iterations, elapsed, cost, best_cost, observation_count,
            rots_rng_draws, (int)requested_signal);
    vector_json(stream, rng, 6);
    fputs(",\"permutation\":", stream); vector_json(stream, p, n);
    fputs(",\"best_permutation\":", stream); vector_json(stream, best, n);
    fputs(",\"tabu\":[", stream);
    for (int i = 0; i < n; ++i) { if (i) fputc(',', stream); vector_json(stream, tabu[i], n); }
    fputs("],\"delta_recovery\":\"Recompute original swap differences from permutation\"}\n",
          stream);
    if (fclose(stream)) exit(73);
}

static int rots_before_iteration(int iteration) {
    (void)iteration;
    return !requested_signal && now_seconds() - search_start < search_limit;
}

static void rots_observe(int iteration, int n, int **a, int **b, int *p, int cost,
                         int *best, int best_cost, int **tabu, int **delta) {
    completed_iterations = iteration;
    if (verify_caches) {
        if (original_cost(n, a, b, p) != cost || original_cost(n, a, b, best) != best_cost) {
            fputs("Original integer objective differs from upstream cache\n", stderr); exit(74);
        }
        for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
            int temporary = p[i]; p[i] = p[j]; p[j] = temporary;
            int64_t difference = original_cost(n, a, b, p) - cost;
            temporary = p[i]; p[i] = p[j]; p[j] = temporary;
            if (difference != delta[i][j]) {
                fputs("Full original swap objective differs from delta cache\n", stderr); exit(75);
            }
        }
    }
    double elapsed = now_seconds() - search_start;
    if (!iteration || best_cost < previous_best) {
        fprintf(observations, "{\"iteration\":%d,\"seconds\":%.17g,\"cost\":%d,"
                "\"permutation\":", iteration, iteration ? elapsed : 0.0, best_cost);
        vector_json(observations, best, n);
        fputs("}\n", observations);
        if (fflush(observations)) exit(76);
        ++observation_count;
        previous_best = best_cost;
    }
    if (!iteration || elapsed - checkpoint_time >= 2.0) {
        checkpoint(iteration ? "periodic" : "initial", n, p, cost, best, best_cost, tabu, elapsed);
        checkpoint_time = now_seconds() - search_start;
    }
}

static void rots_finish(int n, int *p, int cost, int *best, int best_cost, int **tabu) {
    search_seconds = now_seconds() - search_start;
    final_cost = cost; final_best_cost = best_cost;
    memcpy(final_permutation, p, n * sizeof(int));
    memcpy(final_best, best, n * sizeof(int));
    checkpoint(requested_signal ? "interrupted" : "final", n, p, cost,
               best, best_cost, tabu, search_seconds);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fputs("Expected input, seconds, iterations, six comma-separated RNG words, output, verify\n",
              stderr);
        return 2;
    }
    double complete_start = now_seconds();
    search_limit = strtod(argv[2], NULL);
    long iterations = strtol(argv[3], NULL, 10);
    int initial_rng[6];
    if (!(search_limit > 0) || iterations < 1 || iterations > 100000000 ||
        sscanf(argv[4], "%d,%d,%d,%d,%d,%d", initial_rng, initial_rng + 1, initial_rng + 2,
               initial_rng + 3, initial_rng + 4, initial_rng + 5) != 6) return 3;
    for (int i = 0; i < 6; ++i)
        if (initial_rng[i] < 1 || initial_rng[i] >= (i < 3 ? 2147483647 : 2145483479)) return 3;
    evidence_directory = argv[5]; verify_caches = atoi(argv[6]) != 0;
    FILE *input = fopen(argv[1], "r");
    int n = 0;
    if (!input || fscanf(input, "%d", &n) != 1 || n < 2 || n > 2048) return 4;
    int **a = calloc(n, sizeof(*a)), **b = calloc(n, sizeof(*b));
    int *solution = calloc(n, sizeof(int)), *initial = calloc(n, sizeof(int));
    final_permutation = calloc(n, sizeof(int)); final_best = calloc(n, sizeof(int));
    if (!a || !b || !solution || !initial || !final_permutation || !final_best) return 5;
    for (int i = 0; i < n; ++i) {
        a[i] = calloc(n, sizeof(int)); b[i] = calloc(n, sizeof(int));
        if (!a[i] || !b[i]) return 5;
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        if (fscanf(input, "%d", &a[i][j]) != 1 || a[i][j] < 0) return 6;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        if (fscanf(input, "%d", &b[i][j]) != 1 || b[i][j] < 0) return 6;
    fclose(input);
    rng_set(initial_rng);
    generate_random_solution(n, solution);
    int *seen = calloc(n, sizeof(int));
    for (int i = 0; i < n; ++i) {
        if (solution[i] < 0 || solution[i] >= n || seen[solution[i]]++) return 7;
    }
    free(seen);
    memcpy(initial, solution, n * sizeof(int));
    int search_rng[6]; rng_get(search_rng);
    char filename[4096];
    int length = snprintf(filename, sizeof(filename), "%s/initial-state.json", evidence_directory);
    if (length < 0 || (size_t)length >= sizeof(filename)) return 8;
    FILE *initial_file = fopen(filename, "w");
    if (!initial_file) return 8;
    fprintf(initial_file, "{\"cost\":%lld,\"rng_draws\":%llu,\"initial_rng\":",
            (long long)original_cost(n, a, b, initial), rots_rng_draws);
    vector_json(initial_file, initial_rng, 6);
    fputs(",\"search_initial_rng\":", initial_file); vector_json(initial_file, search_rng, 6);
    fputs(",\"permutation\":", initial_file); vector_json(initial_file, initial, n);
    fputs("}\n", initial_file);
    if (fclose(initial_file)) return 8;
    length = snprintf(filename, sizeof(filename), "%s/observations.jsonl", evidence_directory);
    if (length < 0 || (size_t)length >= sizeof(filename)) return 8;
    observations = fopen(filename, "w");
    if (!observations) return 8;
    signal(SIGTERM, interrupted); signal(SIGINT, interrupted);
    opt = -1;
    int best_cost = 0;
    double preparation_seconds = now_seconds() - complete_start;
    search_start = now_seconds();
    tabu_search(n, a, b, solution, &best_cost, 8*n, 5*n*n, (int)iterations);
    if (fclose(observations)) return 9;
    int final_rng[6]; rng_get(final_rng);
    unsigned long long final_draws = rots_rng_draws;
    double probe[64];
    for (int i = 0; i < 64; ++i) probe[i] = rando();
    rng_set(final_rng); rots_rng_draws = final_draws;
    printf("{\"method\":\"taillard-rots-2006\",\"iterations\":%d,\"signal\":%d,"
           "\"search_seconds\":%.17g,\"preparation_seconds\":%.17g,\"rng_draws\":%llu,"
           "\"best_cost\":%d,\"final_cost\":%d,\"initial_cost\":%lld,\"initial_permutation\":",
           completed_iterations, (int)requested_signal, search_seconds, preparation_seconds,
           rots_rng_draws, final_best_cost, final_cost,
           (long long)original_cost(n, a, b, initial));
    vector_json(stdout, initial, n);
    fputs(",\"final_permutation\":", stdout); vector_json(stdout, final_permutation, n);
    fputs(",\"best_permutation\":", stdout); vector_json(stdout, final_best, n);
    fputs(",\"initial_rng\":", stdout); vector_json(stdout, initial_rng, 6);
    fputs(",\"search_initial_rng\":", stdout); vector_json(stdout, search_rng, 6);
    fputs(",\"final_rng\":", stdout); vector_json(stdout, final_rng, 6);
    fputs(",\"next_rng_values\":[", stdout);
    for (int i = 0; i < 64; ++i) printf("%s%.17g", i ? "," : "", probe[i]);
    printf("],\"complete_seconds\":%.17g}\n", now_seconds() - complete_start);
    for (int i = 0; i < n; ++i) { free(a[i]); free(b[i]); }
    free(a); free(b); free(solution); free(initial); free(final_permutation); free(final_best);
    return 0;
}
