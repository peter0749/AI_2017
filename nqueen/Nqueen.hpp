#ifndef __INCLUDE_NQUEEN_HPP
#define __INCLUDE_NQUEEN_HPP
#include "Solver.hpp"
#include <vector>
#include <functional>
/* Interface of N-queen problem:
 * Abstract class: problem(solver, queen_num)
 * Grid size is queen_num x queen_num
 */

class Problem {
    public:
        Problem(void);
        virtual std::pair<int, std::vector<int> > solve(Solver *solver, int queen_num)=0; // solve the n-queen problem and return (#attack, final state)
};

class Nqueen:public Problem {
    public:
        Nqueen(void);
        std::pair<int, std::vector<int> > solve(Solver *solver, int queen_num);
};

#endif
