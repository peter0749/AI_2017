#ifndef __INCLUDE_NQUEEN_HPP
#define __INCLUDE_NQUEEN_HPP
#include "Solver.hpp"
#include <vector>
/* Interface of N-queen problem:
 * Abstract class: problem(solver, queen_num)
 * Grid size is queen_num x queen_num
 */

class Problem {
    public:
        Problem(Solver solver, int queen_num) { }
        virtual std::vector<int> solve(void)=0; // solve the n-queen problem and return #attack
};

class Nqueen:public Problem {
    private:
        Solver solver;
        int queen_n;
    public:
        Nqueen(Solver solver, int queen_num);
        std::vector<int> solve(void);
};

#endif
