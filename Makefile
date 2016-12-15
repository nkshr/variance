all:
	g++ variance.cpp sample.cpp -o variance `pkg-config --cflags opencv` `pkg-config --libs opencv`

clean:
	rm -f *.o
