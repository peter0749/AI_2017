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

Problem::Problem(void) {}

// class Nqueen:
Nqueen::Nqueen(void) {}
std::pair<int, std::vector<int> > Nqueen::solve(Solver *solver, int queen_n) {
    std::vector<int> result = solver->run(queen_n);
    return make_pair(solver->attack_number(result), result);
}

#endif
