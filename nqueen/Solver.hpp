#ifndef __INCLUDE_SOLVER_HPP
#define __INCLUDE_SOLVER_HPP
#include <vector>

/* Interface of solvers */

class Solver {
    public:
        Solver(void);
        Solver(const Solver &ref); // You need to complete copy assianment
        std::vector<int> run(int queen_num);
    protected:
        std::vector<int> state;
};

class HillClimbing:public Solver  {
    public:
        HillClimbing(void);
        HillClimbing(const HillClimbing &ref);
        std::vector<int> run(int queen_num);
};

class GA:public Solver  {
    public:
        GA(void);
        GA(const GA &ref);
        std::vector<int> run(int queen_num);
};
#endif
