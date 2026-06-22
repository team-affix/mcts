GTEST_DIR := googletest
GTEST_INCLUDES := -I$(GTEST_DIR)/googletest/include -I$(GTEST_DIR)/googletest
GTEST_SRCS := $(GTEST_DIR)/googletest/src/gtest-all.cc $(GTEST_DIR)/googletest/src/gtest_main.cc

CXXFLAGS := -g -O2 -std=c++20 -I./include $(GTEST_INCLUDES)
LDFLAGS := -pthread

TEST_BIN := ./build/mcts_test
HEADERS  := $(wildcard include/*.hpp)

all: $(TEST_BIN)

$(TEST_BIN): ./src/mcts_test.cpp $(HEADERS)
	mkdir -p build
	g++ $(CXXFLAGS) $(GTEST_SRCS) ./src/mcts_test.cpp -o $(TEST_BIN) $(LDFLAGS)

test: all
	$(TEST_BIN)

clean:
	rm -rf build
