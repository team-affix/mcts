// uct_cursor holds the node the traversal is currently standing on.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_cursor.hpp"

class UctCursorTest : public ::testing::Test
{
protected:
    monte_carlo::uct_cursor<int> sut{-1};
};

TEST_F(UctCursorTest, StartsAtRoot)
{
    EXPECT_EQ(sut.get_current_node(), -1);
}

TEST_F(UctCursorTest, SetMovesCursor)
{
    sut.set_current_node(4);
    EXPECT_EQ(sut.get_current_node(), 4);
}
