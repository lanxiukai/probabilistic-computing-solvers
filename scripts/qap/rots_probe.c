/* Independent measurement shim for the unchanged official upstream functions.
 * The included external file retains Taillard's non-commercial use conditions.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static int silent_upstream_printf(const char *format, ...) { (void)format; return 0; }
#define printf silent_upstream_printf
#define main unused_upstream_main
#include "upstream.c"
#undef main
#undef printf

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *input = fopen(argv[1], "r");
    int n = 0;
    if (!input || fscanf(input, "%d", &n) != 1 || n < 2 || n > 2048) return 3;
    int **a = calloc(n, sizeof(*a)), **b = calloc(n, sizeof(*b));
    int *p = calloc(n, sizeof(*p)), cost = 0;
    for (int i = 0; i < n; ++i) { a[i] = calloc(n, sizeof(int)); b[i] = calloc(n, sizeof(int)); }
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        if (fscanf(input, "%d", &a[i][j]) != 1) return 4;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        if (fscanf(input, "%d", &b[i][j]) != 1) return 4;
    fclose(input);
    opt = -1;
    generate_random_solution(n, p);
    printf("{\"initial_permutation\":[");
    for (int i = 0; i < n; ++i) printf("%s%d", i ? "," : "", p[i]);
    tabu_search(n, a, b, p, &cost, 8*n, 5*n*n, atoi(argv[2]));
    printf("],\"best_cost\":%d,\"best_permutation\":[", cost);
    for (int i = 0; i < n; ++i) printf("%s%d", i ? "," : "", p[i]);
    printf("],\"next_rng_values\":[");
    for (int i = 0; i < 64; ++i) printf("%s%.17g", i ? "," : "", rando());
    printf("]}\n");
    for (int i = 0; i < n; ++i) { free(a[i]); free(b[i]); }
    free(a); free(b); free(p);
    return 0;
}
