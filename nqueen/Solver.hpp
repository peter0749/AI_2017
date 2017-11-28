#ifndef __INCLUDE_SOLVER_HPP
#define __INCLUDE_SOLVER_HPP
#include <vector>

/* Interface of solvers */

class Solver {
    public:
        Solver(void);
        Solver(const Solver &ref); // You need to complete copy assianment
        std::vector<int> run(int queen_num);
};

class HillClimbing:public Solver  {
    public:
        HillClimbing(void);
        HillClimbing(const HillClimbing &ref);
        std::vector<int> run(int queen_num);
    private:
        std::vector<int> state;
        // --*- HERE_SOME_GREEDY_STUFF -*--
        // HC()?
};

class GA:public Solver  {
    public:
        GA(void);
        GA(const GA &ref);
        std::vector<int> run(int queen_num);
    private:
        // --*- Some member -*--
        // Mating Pool? A vector<> ?
        // Population Poll? A vector<vector< > > ?
        //
        // --*- Some method -*--
        // CrossOver
        // Mutation
        // Selection
        // Survival
        // etc.
};
#endif
