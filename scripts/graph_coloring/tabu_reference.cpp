// Independent TabuCol-style reference; original edge conflicts are maintained exactly.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Int=std::int64_t;
using Clock=std::chrono::steady_clock;
template<class T> void array_json(const std::vector<T>& values) {
    std::cout<<'['; for(std::size_t i=0;i<values.size();++i) { if(i) std::cout<<','; std::cout<<values[i]; } std::cout<<']';
}
int main(int argc,char** argv) {
    if(argc!=9) throw std::runtime_error("Expected graph, colors, seconds, seed, iterations, verify, initial, allowed");
    std::ifstream input(argv[1]);
    int n=0,m=0,k=std::stoi(argv[2]);
    std::string line;
    std::vector<std::pair<int,int>> edges;
    while(std::getline(input,line)) {
        std::istringstream row(line); char type; if(!(row>>type)) continue;
        if(type=='p') { std::string name; row>>name>>n>>m; }
        if(type=='e') { int u,v; row>>u>>v; edges.emplace_back(u-1,v-1); }
    }
    if(n<2 || static_cast<int>(edges.size())!=m || k<2) throw std::runtime_error("Invalid graph header");
    std::vector<std::vector<int>> adj(n);
    for(auto [u,v]:edges) { if(u<0 || u>=n || v<0 || v>=n || u==v) throw std::runtime_error("Invalid graph edge"); adj[u].push_back(v); adj[v].push_back(u); }
    std::vector<int> colors(n),counts(n*k,0),allowed(n*k);
    std::vector<Int> tabu(n*k,0);
    std::ifstream initial_file(argv[7]),allowed_file(argv[8]);
    for(auto& c:colors) initial_file>>c;
    for(auto& a:allowed) allowed_file>>a;
    if(!initial_file || !allowed_file) throw std::runtime_error("Missing initial state or lists");
    for(int i=0;i<n;++i) if(colors[i]<0 || colors[i]>=k || !allowed[i*k+colors[i]]) throw std::runtime_error("Invalid initial color");
    for(int i=0;i<n;++i) for(int j:adj[i]) ++counts[i*k+colors[j]];
    auto objective=[&]() { Int e=0; for(auto [u,v]:edges) e+=colors[u]==colors[v]; return e; };
    double limit=std::stod(argv[3]); Int max_iterations=std::stoll(argv[5]); bool verify=std::stoi(argv[6])!=0;
    std::mt19937_64 rng(std::stoull(argv[4]));
    auto start=Clock::now(); auto elapsed=[&]() { return std::chrono::duration<double>(Clock::now()-start).count(); };
    Int cost=objective(),best_cost=cost,returned_best_cost=cost,iteration=0,quality_iteration=0;
    auto initial=colors,best=colors,returned_best=colors;
    std::vector<Int> curve_cost{cost},curve_iteration{0};
    std::vector<int> curve_eligible{1};
    std::vector<double> curve_seconds{0}; std::vector<std::vector<int>> curve_colors{colors};
    while(cost>0 && iteration<max_iterations && elapsed()<limit) {
        ++iteration; int vertex=-1,color=-1,best_delta=std::numeric_limits<int>::max(); Int ties=0;
        for(int pass=0;pass<2 && vertex<0;++pass) for(int i=0;i<n;++i) if(counts[i*k+colors[i]]>0) for(int c=0;c<k;++c) {
            if(c==colors[i] || !allowed[i*k+c]) continue;
            int delta=counts[i*k+c]-counts[i*k+colors[i]];
            if(!pass && tabu[i*k+c]>=iteration && cost+delta>=returned_best_cost) continue;
            if(delta<best_delta) { vertex=i; color=c; best_delta=delta; ties=1; }
            else if(delta==best_delta && std::uniform_int_distribution<Int>(1,++ties)(rng)==1) { vertex=i; color=c; }
        }
        if(vertex<0) break;
        int old=colors[vertex];
        tabu[vertex*k+old]=iteration+static_cast<int>(0.6*cost)+std::uniform_int_distribution<int>(1,10)(rng);
        for(int j:adj[vertex]) { --counts[j*k+old]; ++counts[j*k+color]; }
        colors[vertex]=color; cost+=best_delta;
        if(verify) {
            if(cost!=objective()) throw std::runtime_error("Conflict cache mismatch");
            for(int i=0;i<n;++i) { std::vector<int> expected(k,0); for(int j:adj[i]) ++expected[colors[j]];
                for(int c=0;c<k;++c) if(counts[i*k+c]!=expected[c]) throw std::runtime_error("Neighbor cache mismatch"); }
        }
        double observed=elapsed(); bool eligible=observed<=limit;
        if(eligible) quality_iteration=iteration;
        if(eligible && cost<best_cost) { best_cost=cost; best=colors; }
        if(cost<returned_best_cost) { returned_best_cost=cost; returned_best=colors; curve_cost.push_back(cost); curve_iteration.push_back(iteration); curve_seconds.push_back(observed); curve_colors.push_back(colors); curve_eligible.push_back(eligible); }
    }
    std::ostringstream rng_state; rng_state<<rng; std::cout.precision(17);
    std::cout<<"{\"best_cost\":"<<best_cost<<",\"returned_best_cost\":"<<returned_best_cost<<",\"final_cost\":"<<cost<<",\"iterations\":"<<iteration<<",\"quality_iteration\":"<<quality_iteration<<",\"search_seconds\":"<<elapsed()<<",\"initial_colors\":";
    array_json(initial); std::cout<<",\"best_colors\":"; array_json(best); std::cout<<",\"final_colors\":"; array_json(colors);
    std::cout<<",\"returned_best_colors\":"; array_json(returned_best);
    std::cout<<",\"curve_cost\":"; array_json(curve_cost); std::cout<<",\"curve_iteration\":"; array_json(curve_iteration);
    std::cout<<",\"curve_seconds\":"; array_json(curve_seconds); std::cout<<",\"curve_eligible\":"; array_json(curve_eligible); std::cout<<",\"curve_colors\":[";
    for(std::size_t i=0;i<curve_colors.size();++i) { if(i) std::cout<<','; array_json(curve_colors[i]); }
    std::cout<<"],\"final_rng\":\""<<rng_state.str()<<"\"}\n";
}
