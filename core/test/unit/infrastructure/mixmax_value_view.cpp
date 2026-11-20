// mixmax_value_view blends a node's reward sum with its best reward:
// (1 - w) * sum + w * visits * best, so the sum over the visits is
// (1 - w) * mean + w * best.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/mixmax_value_view.hpp"

using ::testing::NiceMock;
using ::testing::Return;

namespace
{

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockGetValue
{
    MOCK_METHOD(double, get_value, (const int&), (const));
};

struct MockGetBest
{
    MOCK_METHOD(double, get_value, (const int&), (const));
};

}

class MixmaxValueViewTest : public ::testing::Test
{
protected:
    using view_t = monte_carlo::mixmax_value_view<int, double, MockGetVisits, MockGetValue, MockGetBest>;

    // Node 42: four visits summing to -20 (mean -5), best -2.
    void SetUp() override
    {
        ON_CALL(get_visits, get_visits(42)).WillByDefault(Return(4));
        ON_CALL(get_value, get_value(42)).WillByDefault(Return(-20.0));
        ON_CALL(get_best, get_value(42)).WillByDefault(Return(-2.0));
    }

    view_t make_view(double best_weight)
    {
        return view_t(get_visits, get_value, get_best, best_weight);
    }

    NiceMock<MockGetVisits> get_visits;
    NiceMock<MockGetValue>  get_value;
    NiceMock<MockGetBest>   get_best;
};

TEST_F(MixmaxValueViewTest, NoBestWeightIsThePlainSum)
{
    EXPECT_DOUBLE_EQ(make_view(0.0).get_value(42), -20.0);
}

TEST_F(MixmaxValueViewTest, FullBestWeightIsTheBestOnEveryVisit)
{
    EXPECT_DOUBLE_EQ(make_view(1.0).get_value(42), -8.0);
}

TEST_F(MixmaxValueViewTest, AQuarterBestWeightOverTheVisitsBlendsTheMeanAndTheBest)
{
    // 0.75 * -5 + 0.25 * -2 = -4.25 per visit.
    EXPECT_DOUBLE_EQ(make_view(0.25).get_value(42) / 4.0, -4.25);
}

TEST_F(MixmaxValueViewTest, AnUnvisitedNodeIsWorthNothing)
{
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(0));
    ON_CALL(get_value, get_value(7)).WillByDefault(Return(0.0));
    ON_CALL(get_best, get_value(7)).WillByDefault(Return(0.0));

    EXPECT_DOUBLE_EQ(make_view(0.5).get_value(7), 0.0);
}

TEST_F(MixmaxValueViewTest, EachNodeIsReadOnItsOwnStatistics)
{
    ON_CALL(get_visits, get_visits(7)).WillByDefault(Return(2));
    ON_CALL(get_value, get_value(7)).WillByDefault(Return(6.0));
    ON_CALL(get_best, get_value(7)).WillByDefault(Return(5.0));
    const view_t view = make_view(0.5);

    // Node 7: 0.5 * 6 + 0.5 * 2 * 5 = 8; node 42: 0.5 * -20 + 0.5 * 4 * -2 = -14.
    EXPECT_DOUBLE_EQ(view.get_value(7), 8.0);
    EXPECT_DOUBLE_EQ(view.get_value(42), -14.0);
}
