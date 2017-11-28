#ifndef __INCLUDE_NQUEEN
#define __INCLUDE_NQUEEN
#include "Solver.hpp"
#include "Nqueen.hpp"
#include <vector>
/* Implementation of N-queen problem:
 * Abstract class: problem(solver, queen_num)
 * Grid size is queen_num x queen_num
 */

// class Nqueen:
Nqueen::Nqueen(Solver solver, int queen_num): Problem(solver, queen_num), solver(solver), queen_n(queen_num)  {
    // pass
}
std::vector<int> Nqueen::solve(void) {
    return this->solver.run(queen_n);
}

#endif
