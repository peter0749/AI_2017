all: main
main: main.cpp Nqueen.o Solver.o Solver.hpp Nqueen.hpp
	g++ -std=gnu++11 main.cpp Solver.o Nqueen.o -o main
Nqueen.o: Solver.hpp Nqueen.hpp Nqueen.cpp
	g++ -c -std=gnu++11 Nqueen.cpp -o Nqueen.o
Solver.o: Solver.hpp Solver.cpp
	g++ -c -std=gnu++11 Solver.cpp -o Solver.o
clean:
	rm -f *.o main
