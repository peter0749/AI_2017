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
    if (argc<2) return 1;
    int n_queen = atoi(argv[1]);
    Nqueen nq;
    HillClimbing HC;
    // GA ga; // not implment yet
    pair<int, vector<int> > result = nq.solve(&HC, n_queen);
    for (int i=0; i<n_queen; ++i) {
        int j=0;
        for (; j<result.second[i]; ++j) cout << "0 ";
        cout << "1 "; 
        for (++j; j<n_queen; ++j) cout << "0 ";
        cout << endl;
    }
    cout << "#attack: " << result.first << endl;
    return 0;
}
