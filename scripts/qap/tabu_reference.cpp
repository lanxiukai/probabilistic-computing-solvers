// Independently implemented attribute tabu reference with exact general-QAP deltas.
// The method is inspired by Taillard's robust tabu search, not an upstream replica.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

using Int = std::int64_t;
using Clock = std::chrono::steady_clock;

template<class T> void array_json(const std::vector<T>& values, std::ostream& stream=std::cout) {
    stream << '[';
    for (std::size_t i=0;i<values.size();++i) { if(i) stream<<','; stream<<values[i]; }
    stream << ']';
}

int main(int argc,char** argv) {
    if(argc!=7) throw std::runtime_error("Expected instance, seconds, seed, iterations, verify, initial");
    std::ifstream input(argv[1]);
    int n; input>>n;
    if(!input || n<2 || n>2048) throw std::runtime_error("Invalid QAP order");
    std::vector<Int> f(n*n),d(n*n),delta(n*n),tabu(n*n,0);
    for(auto& v:f) input>>v;
    for(auto& v:d) input>>v;
    if(!input) throw std::runtime_error("Incomplete QAP matrices");
    double limit=std::stod(argv[2]);
    std::uint64_t seed=std::stoull(argv[3]);
    Int max_iterations=std::stoll(argv[4]);
    bool verify=std::stoi(argv[5])!=0;
    std::mt19937_64 rng(seed);
    std::vector<int> p(n);
    std::ifstream initial_file(argv[6]);
    for(auto& v:p) initial_file>>v;
    auto sorted=p; std::sort(sorted.begin(),sorted.end());
    for(int i=0;i<n;++i) if(!initial_file || sorted[i]!=i) throw std::runtime_error("Invalid initial permutation");
    auto objective=[&]() { Int value=0; for(int i=0;i<n;++i) for(int j=0;j<n;++j) value+=f[i*n+j]*d[p[i]*n+p[j]]; return value; };
    auto difference=[&](int i,int j) {
        Int value=(f[i*n+i]-f[j*n+j])*(d[p[j]*n+p[j]]-d[p[i]*n+p[i]])
            +(f[i*n+j]-f[j*n+i])*(d[p[j]*n+p[i]]-d[p[i]*n+p[j]]);
        for(int k=0;k<n;++k) if(k!=i && k!=j) {
            value+=(f[i*n+k]-f[j*n+k])*(d[p[j]*n+p[k]]-d[p[i]*n+p[k]])
                +(f[k*n+i]-f[k*n+j])*(d[p[k]*n+p[j]]-d[p[k]*n+p[i]]);
        }
        return value;
    };
    auto reset_delta=[&]() { for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) delta[i*n+j]=difference(i,j); };
    auto started=Clock::now();
    auto elapsed=[&]() { return std::chrono::duration<double>(Clock::now()-started).count(); };
    Int cost=objective(),best_cost=cost;
    auto initial=p,best=p;
    std::vector<Int> curve_cost{best_cost},curve_iteration{0};
    std::vector<double> curve_seconds{0};
    std::vector<std::vector<int>> curve_permutations{p};
    reset_delta();
    Int iteration=0,last_improvement=0,restarts=0;
    std::ofstream progress(std::string(argv[6])+".progress.jsonl");
    if(!progress) throw std::runtime_error("Cannot write reference recovery history");
    progress.precision(17);
    auto checkpoint=[&](const char* kind) {
        std::ostringstream saved_rng; saved_rng<<rng;
        progress<<"{\"kind\":\""<<kind<<"\",\"iteration\":"<<iteration
            <<",\"seconds\":"<<elapsed()<<",\"cost\":"<<cost<<",\"best_cost\":"<<best_cost
            <<",\"last_improvement\":"<<last_improvement<<",\"restarts\":"<<restarts<<",\"permutation\":";
        array_json(p,progress); progress<<",\"best_permutation\":"; array_json(best,progress);
        progress<<",\"rng\":\""<<saved_rng.str()<<"\"}\n"; progress.flush();
    };
    checkpoint("initial");
    while(iteration<max_iterations && elapsed()<limit) {
        ++iteration;
        int r=-1,s=-1;
        Int best_delta=std::numeric_limits<Int>::max(),ties=0;
        for(int pass=0;pass<2 && r<0;++pass) for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) {
            Int value=delta[i*n+j];
            bool allowed=pass || (tabu[i*n+p[j]]<iteration && tabu[j*n+p[i]]<iteration) || cost+value<best_cost;
            if(!allowed) continue;
            if(value<best_delta) { best_delta=value; r=i; s=j; ties=1; }
            else if(value==best_delta && std::uniform_int_distribution<Int>(1,++ties)(rng)==1) { r=i; s=j; }
        }
        int old_r=p[r],old_s=p[s];
        cost+=best_delta;
        std::swap(p[r],p[s]);
        int tenure=std::uniform_int_distribution<int>(std::max(1,4*n/5),std::max(2,6*n/5))(rng);
        tabu[r*n+old_r]=iteration+tenure;
        tabu[s*n+old_s]=iteration+tenure;
        for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) {
            if(i==r || i==s || j==r || j==s) delta[i*n+j]=difference(i,j);
            else delta[i*n+j]+=(f[i*n+r]-f[j*n+r]-f[i*n+s]+f[j*n+s])*
                (d[p[j]*n+old_s]-d[p[i]*n+old_s]-d[p[j]*n+old_r]+d[p[i]*n+old_r])
                +(f[r*n+i]-f[r*n+j]-f[s*n+i]+f[s*n+j])*
                (d[old_s*n+p[j]]-d[old_s*n+p[i]]-d[old_r*n+p[j]]+d[old_r*n+p[i]]);
        }
        if(verify) {
            if(cost!=objective()) throw std::runtime_error("Objective cache mismatch");
            for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) if(delta[i*n+j]!=difference(i,j)) throw std::runtime_error("Delta cache mismatch");
        }
        if(cost<best_cost) {
            best_cost=cost; best=p; last_improvement=iteration;
            curve_cost.push_back(cost); curve_seconds.push_back(elapsed()); curve_iteration.push_back(iteration); curve_permutations.push_back(p);
            checkpoint("improved");
        }
        if(iteration-last_improvement>5*n*n) {
            std::shuffle(p.begin(),p.end(),rng); cost=objective(); reset_delta();
            std::fill(tabu.begin(),tabu.end(),0); last_improvement=iteration; ++restarts;
            if(cost<best_cost) {
                best_cost=cost; best=p;
                curve_cost.push_back(cost); curve_seconds.push_back(elapsed()); curve_iteration.push_back(iteration); curve_permutations.push_back(p);
            }
            checkpoint("restart");
        }
        if(iteration%1024==0) checkpoint("periodic");
    }
    double seconds=elapsed();
    checkpoint("final");
    std::ostringstream rng_state; rng_state<<rng;
    std::cout.precision(17);
    std::cout<<"{\"best_cost\":"<<best_cost<<",\"final_cost\":"<<cost<<",\"iterations\":"<<iteration
        <<",\"search_seconds\":"<<seconds<<",\"restarts\":"<<restarts<<",\"initial_permutation\":";
    array_json(initial); std::cout<<",\"best_permutation\":"; array_json(best);
    std::cout<<",\"final_permutation\":"; array_json(p);
    std::cout<<",\"curve_cost\":"; array_json(curve_cost);
    std::cout<<",\"curve_iteration\":"; array_json(curve_iteration);
    std::cout<<",\"curve_seconds\":"; array_json(curve_seconds);
    std::cout<<",\"curve_permutations\":[";
    for(std::size_t i=0;i<curve_permutations.size();++i) { if(i) std::cout<<','; array_json(curve_permutations[i]); }
    std::cout<<"],\"final_rng\":\""<<rng_state.str()<<"\"}\n";
}
