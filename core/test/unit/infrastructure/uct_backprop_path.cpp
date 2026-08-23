// uct_backprop_path is a stack seeded with the root, so draining it walks the
// episode leaf-to-root.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/uct_backprop_path.hpp"

class UctBackpropPathTest : public ::testing::Test
{
protected:
    monte_carlo::uct_backprop_path<int> sut{-1};
};

TEST_F(UctBackpropPathTest, SeededWithRoot)
{
    EXPECT_EQ(sut.size(), 1u);
    EXPECT_EQ(sut.top(), -1);
}

TEST_F(UctBackpropPathTest, PushThenPopUnwindsInReverseOrder)
{
    sut.push(0);
    sut.push(1);

    EXPECT_EQ(sut.size(), 3u);

    EXPECT_EQ(sut.top(), 1);
    sut.pop();
    EXPECT_EQ(sut.top(), 0);
    sut.pop();
    EXPECT_EQ(sut.top(), -1);
    sut.pop();

    EXPECT_EQ(sut.size(), 0u);
}
