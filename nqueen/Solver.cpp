#ifndef __INCLUDE_SOLVER
#define __INCLUDE_SOLVER
#include <vector>
#include <cmath>
#include <climits>
#include <algorithm>
#include <map>
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
GA::GA(size_t state_size, size_t polulation_size, unsigned int tournament, \
       double cross_over_rate, double mutation_rate, \
       unsigned int termination, unsigned int runs \
      ): state_size(state_size), polulation_size(polulation_size), tournament(tournament), cx_r(cross_over_rate), mutation_r(mutation_rate), termination(termination), runs(runs) {
    if (this->mutation_r<0) this->mutation_r = 1. / (double)state_size;
}

GA::GA(const GA &ref): state_size(ref.state_size), polulation_size(ref.polulation_size), tournament(ref.tournament), cx_r(ref.cx_r), mutation_r(ref.mutation_r), termination(ref.termination), runs(ref.runs) {}

std::pair<std::vector<int>, std::vector<int> > GA::CrossOver(const std::vector<int> &p,
                               const std::vector<int> &q) {
    if ((double)rand()/(double)RAND_MAX > this->cx_r) return make_pair(p,q);
    std::vector<int> child_1(p.size(), 0), child_2(p.size(), 0);
    int l = rand()%p.size();
    int r = rand()%p.size(); // 2-point CX
    if (l>r) std::swap(l,r);
    else if (l==r) ++r; // [l, r)
    std::map<int,int> mapping_pq, mapping_qp; // PMX
    for (int i=l; i<r; ++i) {
        mapping_pq[p[i]] = q[i];
        mapping_qp[q[i]] = p[i];
        child_1[i] = p[i];
        child_2[i] = q[i];
    }
    for (int i=0; i<l; ++i) {
        child_1[i]=q[i], child_2[i]=p[i];
        while(mapping_pq.count(child_1[i])) child_1[i] = mapping_pq[child_1[i]];
        while(mapping_qp.count(child_2[i])) child_2[i] = mapping_qp[child_2[i]];
    }
    for (int i=r; i<p.size(); ++i) {
        child_1[i]=q[i], child_2[i]=p[i];
        while(mapping_pq.count(child_1[i])) child_1[i] = mapping_pq[child_1[i]];
        while(mapping_qp.count(child_2[i])) child_2[i] = mapping_qp[child_2[i]];
    }

    return make_pair(child_1, child_2);
}

void GA::Mutation(std::vector<int> &gene) {
    if ((double)rand()/(double)RAND_MAX < this->mutation_r) {
        int l=rand()%gene.size();
        int r=rand()%gene.size();
        std::swap(gene[l],gene[r]);
    }
}

std::vector<int> GA::Selection(const int sel_n) {
    std::vector<int> selected_index;
    while(selected_index.size()<sel_n) {
        for (int i=0; i<this->polulationPool.size(); ++i) swap(this->polulationPool[i], this->polulationPool[rand()%this->polulationPool.size()]); // shuffle
        int best_score = this->attack_number(this->polulationPool[0]); 
        int best_idx = 0;
        for (int i=std::min((int)this->tournament, (int)this->polulationPool.size())-1; i>0; --i) {
            int score = this->attack_number(this->polulationPool[i]);
            if (score<best_score) {
                best_score = score;
                best_idx = i;
            }
        }
        selected_index.push_back(best_idx);
    }
    return selected_index;
}

bool cmp(const std::vector<int> &a, const std::vector<int> &b) {
    return Solver::attack_number(a) < Solver::attack_number(b);
}

void GA::Survival(const int survive_n) {
    sort(this->polulationPool.begin(), this->polulationPool.end(), cmp);
    this->polulationPool.resize(survive_n); // truncated
}

std::vector<int> GA::run(int queen_num) {
    std::vector<int> best_run;
    this->random_init(this->state_size, best_run);
    int best_score = this->attack_number(best_run);
    for (int R=0; R<this->runs; ++R) {
        // some stuff
        for (int i=0; i<this->polulation_size; ++i) {
            std::vector<int> individual;
            this->random_init(this->state_size, individual);
            this->polulationPool.push_back(individual);
        }
        for (int T=0; T<this->termination; ++T) {
            std::vector<int> selected = this->Selection(this->polulation_size); // selection
            for (int i=0; i+1<selected.size(); i+=2) {
                std::pair<std::vector<int>, std::vector<int> > childs = \
                    this->CrossOver(this->polulationPool[selected[i]], this->polulationPool[selected[i+1]]);
                this->Mutation(childs.first);
                this->Mutation(childs.second);
                this->polulationPool.push_back(childs.first);
                this->polulationPool.push_back(childs.second); // mu+lambda
            }
            this->Survival(this->polulation_size);
            int top_score = this->attack_number(this->polulationPool[0]);
            if (top_score<best_score) {
                best_score = top_score;
                best_run = this->polulationPool[0];
            }
            // some stuff
        }
        this->polulationPool.clear();
    }
    return best_run;
}

#endif
