CXX = g++
CXXFLAGS = --std=c++11 -Wall -Wno-sign-compare -Wno-unknown-pragmas -fPIC -fopenmp -O3 -pthread
INCLUDES = -I"/usr/include/"

all: w2v-sembei

w2v-sembei: w2v-sembei.cpp docopt.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) docopt.o w2v-sembei.cpp -o w2v-sembei
docopt.o: docopt.cpp/docopt.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c docopt.cpp/docopt.cpp -o docopt.o
check-syntax:
	$(CXX) -o nul $(CXXFLAGS) $(INCLUDES) -S w2v-sembei.cpp

.PHONY : clean

clean:
	rm *.o w2v-sembei
