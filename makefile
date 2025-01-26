all:
	mkdir -p build
	g++ -g -std=c++20 -I"./include/" $(wildcard ./src/*) -o ./build/main
	
clean:
	rm -rf build
	