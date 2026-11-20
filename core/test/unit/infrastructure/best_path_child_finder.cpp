// best_path_child_finder returns the best episode's choice at a node, or
// nothing where that episode did not pass.

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/best_path_child_finder.hpp"

using ::testing::NiceMock;
using ::testing::Return;

namespace
{

struct stub_choice_count
{
    size_t size() const { return 2; }
};

struct stub_choice_at
{
    int at(size_t i) const { return static_cast<int>(i); }
};

struct MockFindStep
{
    MOCK_METHOD(std::optional<int>, find_step, (const int&), (const));
};

}

class BestPathChildFinderTest : public ::testing::Test
{
protected:
    NiceMock<MockFindStep> find_step;
    monte_carlo::best_path_child_finder<int, int, MockFindStep, stub_choice_count, stub_choice_at> sut{find_step};
    stub_choice_count count;
    stub_choice_at    at;
};

TEST_F(BestPathChildFinderTest, ReturnsTheBestPathsChoiceAtTheNode)
{
    ON_CALL(find_step, find_step(4)).WillByDefault(Return(std::optional<int>(1)));
    EXPECT_EQ(sut.find(4, count, at), std::optional<int>(1));
}

TEST_F(BestPathChildFinderTest, FindsNothingOffTheBestPath)
{
    ON_CALL(find_step, find_step(4)).WillByDefault(Return(std::nullopt));
    EXPECT_EQ(sut.find(4, count, at), std::nullopt);
}
