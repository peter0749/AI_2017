all: main
main: main.cpp Nqueen.o Solver.o Solver.hpp Nqueen.hpp
	g++ -std=gnu++11 main.cpp Solver.o Nqueen.o -o main -O2
Nqueen.o: Solver.hpp Nqueen.hpp Nqueen.cpp
	g++ -c -std=gnu++11 Nqueen.cpp -o Nqueen.o -O2
Solver.o: Solver.hpp Solver.cpp
	g++ -c -std=gnu++11 Solver.cpp -o Solver.o -O2
clean:
	rm -f *.o main
