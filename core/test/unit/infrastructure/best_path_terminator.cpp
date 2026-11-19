// best_path_terminator grafts the episode's path, then terminates as UCT does.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/best_path_terminator.hpp"

using ::testing::InSequence;
using ::testing::StrictMock;

namespace
{

struct MockGraft
{
    MOCK_METHOD(void, graft, (), ());
};

struct MockTerminate
{
    MOCK_METHOD(void, terminate, (), ());
};

}

class BestPathTerminatorTest : public ::testing::Test
{
protected:
    StrictMock<MockGraft>     grafter;
    StrictMock<MockTerminate> terminator;
    monte_carlo::best_path_terminator<MockGraft, MockTerminate> sut{grafter, terminator};
};

TEST_F(BestPathTerminatorTest, GraftsBeforeTerminating)
{
    InSequence order;
    EXPECT_CALL(grafter, graft());
    EXPECT_CALL(terminator, terminate());

    sut.terminate();
}

TEST_F(BestPathTerminatorTest, EachTerminationGraftsAndTerminatesOnce)
{
    InSequence order;
    EXPECT_CALL(grafter, graft());
    EXPECT_CALL(terminator, terminate());
    EXPECT_CALL(grafter, graft());
    EXPECT_CALL(terminator, terminate());

    sut.terminate();
    sut.terminate();
}
