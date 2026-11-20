// focus_ucb1 scores children as ucb1 does, plus a bonus for the focus: the
// first child, in index order, whose normalized mean is within the band of the
// best. Unvisited children are chosen first.

#include <cstddef>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/focus_ucb1.hpp"
#include "infrastructure/uniform_exploration_constant.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;

namespace
{

struct stub_choice_count
{
    size_t size() const { return 3; }
};

struct stub_choice_at
{
    int at(size_t i) const { return static_cast<int>(i); }
};

struct MockGetVisits
{
    MOCK_METHOD(size_t, get_visits, (const int&), (const));
};

struct MockGetValue
{
    MOCK_METHOD(double, get_value, (const int&), (const));
};

struct MockIncludeValue
{
    MOCK_METHOD(void, include, (double), ());
};

struct MockNormalizeValue
{
    MOCK_METHOD(double, normalize, (double), (const));
};

struct MockWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

struct MockGetFocusBand
{
    MOCK_METHOD(double, get_band, (size_t), (const));
};

struct MockGetFocusBonus
{
    MOCK_METHOD(double, get_focus_bonus, (const int&), (const));
};

double unchanged(double mean)
{
    return mean;
}

using exploration_t = monte_carlo::uniform_exploration_constant<double>;
using policy_t      = monte_carlo::focus_ucb1<int, int, double, MockGetVisits, MockGetValue, MockIncludeValue,
                                              MockNormalizeValue, MockWalker, exploration_t, MockGetFocusBand,
                                              MockGetFocusBonus, stub_choice_count, stub_choice_at>;

}

class FocusUcb1Test : public ::testing::Test
{
protected:
    static constexpr int parent = 99;

    NiceMock<MockGetVisits>      get_visits;
    NiceMock<MockGetValue>       get_value;
    NiceMock<MockIncludeValue>   include_value;
    NiceMock<MockNormalizeValue> normalize_value;
    NiceMock<MockWalker>         walker;
    NiceMock<MockGetFocusBand>   get_focus_band;
    NiceMock<MockGetFocusBonus>  get_focus_bonus;
    exploration_t                no_exploration{0.0};
    exploration_t                unit_exploration{1.0};
    policy_t greedy{get_visits, get_value, include_value, normalize_value, walker, no_exploration, get_focus_band,
                    get_focus_bonus};
    policy_t exploring{get_visits, get_value, include_value, normalize_value, walker, unit_exploration,
                       get_focus_band, get_focus_bonus};
    stub_choice_count count;
    stub_choice_at    at;

    void SetUp() override
    {
        for (int child = 0; child < 3; ++child)
            ON_CALL(walker, walk(parent, child)).WillByDefault(Return(child));
        ON_CALL(normalize_value, normalize(_)).WillByDefault(&unchanged);
        ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(30));
    }

    // The three children's means, each child with `visits` visits.
    void set_means(double first, double second, double third, size_t visits)
    {
        const double means[] = {first, second, third};
        for (int child = 0; child < 3; ++child)
        {
            ON_CALL(get_visits, get_visits(child)).WillByDefault(Return(visits));
            ON_CALL(get_value, get_value(child)).WillByDefault(Return(means[child] * static_cast<double>(visits)));
        }
    }

    void set_focus(double band, double bonus)
    {
        ON_CALL(get_focus_band, get_band(_)).WillByDefault(Return(band));
        ON_CALL(get_focus_bonus, get_focus_bonus(_)).WillByDefault(Return(bonus));
    }
};

TEST_F(FocusUcb1Test, AnUnvisitedChildIsChosenFirst)
{
    set_means(0.9, 0.1, 0.1, 10);
    ON_CALL(get_visits, get_visits(2)).WillByDefault(Return(0));
    EXPECT_CALL(get_focus_band, get_band(_)).Times(0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 2);
}

TEST_F(FocusUcb1Test, WithoutABonusTheHighestMeanWins)
{
    set_means(0.2, 0.7, 0.5, 10);
    set_focus(0.5, 0.0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 1);
}

TEST_F(FocusUcb1Test, TheFirstChildWithinTheBandOfTheBestGetsTheBonus)
{
    // 0.6 - 0.5 is within the band 0.15, so child 0 scores 0.5 + 0.2 against 0.6.
    set_means(0.5, 0.6, 0.6, 10);
    set_focus(0.15, 0.2);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 0);
}

TEST_F(FocusUcb1Test, OnlyTheFirstNearBestChildIsTheFocus)
{
    // Children 1 and 2 are both within the band; only child 1 gets the bonus, so it wins though child 2's mean is
    // higher.
    set_means(0.2, 0.6, 0.62, 10);
    set_focus(0.05, 0.1);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 1);
}

TEST_F(FocusUcb1Test, AChildOutsideTheBandGetsNoBonus)
{
    // Child 0 is 0.2 below the best, outside the band 0.1, so its bonus would have made it win but it gets none.
    set_means(0.4, 0.6, 0.1, 10);
    set_focus(0.1, 0.25);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 1);
}

TEST_F(FocusUcb1Test, TheBestChildIsTheFocusWhenNoEarlierChildIsNearIt)
{
    set_means(0.1, 0.2, 0.9, 10);
    set_focus(0.05, 0.3);
    ON_CALL(get_visits, get_visits(2)).WillByDefault(Return(1000));
    ON_CALL(get_value, get_value(2)).WillByDefault(Return(900.0));

    // Child 2's bonus outweighs the others' larger exploration terms.
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(1020));
    EXPECT_EQ(exploring.policy_choose(parent, count, at), 2);
}

TEST_F(FocusUcb1Test, TheBandIsReadForTheParentsVisits)
{
    set_means(0.2, 0.6, 0.5, 10);
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(17));
    EXPECT_CALL(get_focus_band, get_band(17)).WillOnce(Return(0.1));

    greedy.policy_choose(parent, count, at);
}

TEST_F(FocusUcb1Test, ExplorationStillOutweighsTheBonusEventually)
{
    // Equal means; the focus has 1,000 visits and its sibling 1. Bonus 0.5 against sqrt(ln 1002 / 1) ~ 2.6.
    set_means(0.6, 0.6, 0.0, 1000);
    ON_CALL(get_visits, get_visits(1)).WillByDefault(Return(1));
    ON_CALL(get_value, get_value(1)).WillByDefault(Return(0.6));
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(2002));
    set_focus(0.1, 0.5);

    EXPECT_EQ(exploring.policy_choose(parent, count, at), 1);
}

TEST_F(FocusUcb1Test, TheBonusOutweighsASmallExplorationEdge)
{
    // Equal means and visits but for a slight edge in child 1's exploration; the focus's bonus wins.
    set_means(0.6, 0.6, 0.0, 100);
    ON_CALL(get_visits, get_visits(1)).WillByDefault(Return(90));
    ON_CALL(get_value, get_value(1)).WillByDefault(Return(0.6 * 90));
    ON_CALL(get_visits, get_visits(parent)).WillByDefault(Return(290));
    set_focus(0.1, 0.5);

    EXPECT_EQ(exploring.policy_choose(parent, count, at), 0);
}

TEST_F(FocusUcb1Test, EveryMeanIsRecordedBeforeAnyIsNormalized)
{
    set_means(0.2, 0.6, 0.5, 10);
    set_focus(0.1, 0.1);

    InSequence order;
    EXPECT_CALL(include_value, include(_)).Times(3);
    EXPECT_CALL(normalize_value, normalize(_)).Times(AnyNumber());
    greedy.policy_choose(parent, count, at);
}

TEST_F(FocusUcb1Test, TheBestIsTheNormalizedHighestMean)
{
    set_means(0.2, 0.6, 0.5, 10);
    set_focus(0.1, 0.1);
    EXPECT_CALL(normalize_value, normalize(_)).Times(AnyNumber());
    EXPECT_CALL(normalize_value, normalize(0.6)).Times(::testing::AtLeast(1)).WillRepeatedly(&unchanged);

    greedy.policy_choose(parent, count, at);
}

TEST_F(FocusUcb1Test, ATieInScoreGoesToTheFirstChild)
{
    set_means(0.6, 0.6, 0.6, 10);
    set_focus(0.0, 0.0);

    EXPECT_EQ(greedy.policy_choose(parent, count, at), 0);
}
