#include <iostream>
#include <vector>
#include <functional>
#include <cstdlib>
#include <ctime>
#include "Solver.hpp"
#include "Nqueen.hpp"

using namespace std;

int main(int argc, char **argv) {
    srand(time(NULL));
    if (argc<3) {
        cerr << "usage: ./" << argv[0] << "nqueen round" << endl;
        return 1;
    }
    int n_queen = atoi(argv[1]);
    int round = atoi(argv[2]);
    Nqueen nq;
    HillClimbing HC;
    GA ga(n_queen, 100, 5, 0.95, 0.95, 800, 1);
    int ga_succ=0, hc_succ=0;
    int ga_tatk=0, hc_tatk=0;
    for (int i=0; i<round; ++i) {
        pair<int, vector<int> > result = nq.solve(&HC, n_queen);
        cout << "HC:" << endl;
        for (const auto v: result.second) cout << v << ' ';
        cout << "\n#attack: " << result.first << '\n' << endl;
        hc_tatk += result.first;
        hc_succ += (result.first==0);

        cout << "GA:" << endl;
        for (const auto v: result.second) cout << v << ' ';
        result = nq.solve(&ga, n_queen);
        cout << "\n#attack: " << result.first << '\n' << endl;
        ga_tatk += result.first;
        ga_succ += (result.first==0);
    }
    cout << "HC:" << endl;
    cout << "average #attack: " << (float)hc_tatk/round << endl;
    cout << "success rate: " << (float)hc_succ/round << '\n' << endl;

    cout << "GA:" << endl;
    cout << "average #attack: " << (float)ga_tatk/round << endl;
    cout << "success rate: " << (float)ga_succ/round << endl;

    return 0;
}
