#ifndef __INCLUDE_SOLVER_HPP
#define __INCLUDE_SOLVER_HPP
#include <vector>
#include <cmath>
#include <climits>
#include <algorithm>

/* Interface of solvers */

class Solver {
    public:
        Solver(void);
        Solver(const Solver &ref); // You need to complete copy assianment
        virtual std::vector<int> run(int queen_num)=0;
        static int attack_number(const std::vector<int> state);
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
        GA(size_t state_size, size_t polulation_size=300, unsigned int tournament=5, \
           double cross_over_rate=0.9, double mutation_rate=-1, \
           unsigned int termination=100, unsigned int runs=30
          );
        GA(const GA &ref);
        virtual std::vector<int> run(int queen_num); // output final state
    private:
        // --*- Some member -*--
        const size_t state_size;
        double mutation_r, cx_r; // rate
        unsigned int tournament; // for selection
        const size_t polulation_size;
        const unsigned int termination;
        const unsigned int runs;
        std::vector<int> matingPool; // selected index of parents
        std::vector<std::vector<int> > polulationPool;
        // --*- Some method -*--
        std::pair<std::vector<int>, std::vector<int> > CrossOver(const std::vector<int> &p, 
                                   const std::vector<int> &q);
        void Mutation(std::vector<int> &gene);
        // Selection: return index of selected gene
        std::vector<int> Selection(const int sel_n); 
        // End Selection.
        void Survival(const int survive_n); // next gen.
};
#endif
