#ifndef __INCLUDE_SOLVER
#define __INCLUDE_SOLVER
#include <vector>
#include <cmath>
#include <climits>
#include <algorithm>
#include "Solver.hpp"
using std::abs;

/* Implementation of solvers */

// class Solver:
Solver::Solver(void) {}
Solver::Solver(const Solver &ref) {}

void Solver::random_init(int n, std::vector<int> &state) {
    state.resize(n);
    for (int i=0; i<n; ++i) state[i]=i;
    for (int i=0; i<n; ++i) std::swap(state[i], state[rand()%n]); // shuffle
}
int Solver::attack_number(const std::vector<int> state) {
    int attack=0;
    for (int i=1; i<state.size(); ++i) {
        for (int j=0; j<i; ++j) {
            int u=state[i], v=state[j];
            if (abs(u-v)==i-j) ++attack;
        }
    }
    return attack; 
}

// class HillClimbing:
HillClimbing::HillClimbing(void) {};
HillClimbing::HillClimbing(const HillClimbing &ref) {};
std::vector<int> HillClimbing::run(int queen_num) {
    this->random_init(queen_num, this->state);
    int best_score = this->attack_number(this->state);
    for (;;) {
        int score = this->to_best_neighbor();
        if (score>=best_score) return this->state; // no improvement. halt.
        best_score = score;
    }
}
int HillClimbing::to_best_neighbor(void) {
    int best_i=0, best_j=0;
    int best_score=this->attack_number(state);
    for (int i=0; i<state.size(); ++i) {
        for (int j=i+1; j<state.size(); ++j) {
            std::swap(state[i], state[j]);
            int score = this->attack_number(state);
            if (score<best_score) {
                best_score = score;
                best_i = i, best_j = j;
            }
            std::swap(state[i], state[j]);
        }
    }
    std::swap(state[best_i], state[best_j]);
    return best_score;
}

// class GA:
// pass

#endif
