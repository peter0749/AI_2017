#ifndef __INCLUDE_NQUEEN
#define __INCLUDE_NQUEEN
#include "Solver.hpp"
#include "Nqueen.hpp"
#include <vector>
#include <functional>
/* Implementation of N-queen problem:
 * Abstract class: problem(solver, queen_num)
 * Grid size is queen_num x queen_num
 */

Problem::Problem(Solver *s, int q) {}
// class Nqueen:
Nqueen::Nqueen(Solver *solver, int queen_num): Problem(solver, queen_num), solver(solver), queen_n(queen_num)  {
    // pass
}
std::pair<int, std::vector<int> > Nqueen::solve(void) {
    std::vector<int> result = this->solver->run(this->queen_n);
    return make_pair(this->solver->attack_number(result), result);
}

#endif
