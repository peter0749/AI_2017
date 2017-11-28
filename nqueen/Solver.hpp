#ifndef __INCLUDE_SOLVER_HPP
#define __INCLUDE_SOLVER_HPP
#include <vector>
#include <cmath>
#include <climits>
#include <algorithm>
using std::abs;

/* Interface of solvers */

class Solver {
    public:
        Solver(void);
        Solver(const Solver &ref); // You need to complete copy assianment
        virtual std::vector<int> run(int queen_num)=0;
        int attack_number(const std::vector<int> state);
    protected:
        void random_init(int n, std::vector<int> &state);
};

class HillClimbing:public Solver  {
    public:
        HillClimbing(void);
        HillClimbing(const HillClimbing &ref);
        virtual std::vector<int> run(int queen_num);
    private:
        std::vector<int> state;
        int to_best_neighbor(void);
};

class GA:public Solver  {
    public:
        GA(void);
        GA(const GA &ref);
        virtual std::vector<int> run(int queen_num);
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
