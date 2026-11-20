// best_path_recorder replaces the best path with the episode's when its reward
// beats the best so far, walking the logged choices from the root and storing
// the choice made at each node; the log is cleared either way.

#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/best_path_recorder.hpp"

using ::testing::_;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

namespace
{

struct MockGetReward
{
    MOCK_METHOD(double, value, (), (const));
};

struct MockGetChoices
{
    MOCK_METHOD(const std::vector<int>&, choices, (), (const));
};

struct MockClearChoices
{
    MOCK_METHOD(void, clear_choices, (), ());
};

struct MockGetBestReward
{
    MOCK_METHOD(double, best_reward, (), (const));
};

struct MockStartPath
{
    MOCK_METHOD(void, start_path, (double), ());
};

struct MockAddStep
{
    MOCK_METHOD(void, add_step, (const int&, const int&), ());
};

struct MockWalker
{
    MOCK_METHOD(int, walk, (const int&, int), (const));
};

using recorder_t = monte_carlo::best_path_recorder<int, int, double, MockGetReward, MockGetChoices,
                                                   MockClearChoices, MockGetBestReward, MockStartPath,
                                                   MockAddStep, MockWalker>;

}

class BestPathRecorderTest : public ::testing::Test
{
protected:
    static constexpr int kRoot = 0;

    NiceMock<MockGetReward>     get_reward;
    NiceMock<MockGetChoices>    get_choices;
    NiceMock<MockClearChoices>  clear_choices;
    NiceMock<MockGetBestReward> get_best_reward;
    NiceMock<MockStartPath>     start_path;
    NiceMock<MockAddStep>       add_step;
    NiceMock<MockWalker>        walker;
    recorder_t sut{get_reward, get_choices, clear_choices, get_best_reward, start_path, add_step, walker, kRoot};

    std::vector<int> choices{1, 0, 1};

    void SetUp() override
    {
        ON_CALL(get_choices, choices()).WillByDefault(ReturnRef(choices));
        ON_CALL(get_best_reward, best_reward()).WillByDefault(Return(-10.0));
        ON_CALL(walker, walk(kRoot, 1)).WillByDefault(Return(5));
        ON_CALL(walker, walk(5, 0)).WillByDefault(Return(8));
        ON_CALL(walker, walk(8, 1)).WillByDefault(Return(9));
    }
};

TEST_F(BestPathRecorderTest, ABetterEpisodeReplacesTheBestPathStepByStepFromTheRoot)
{
    ON_CALL(get_reward, value()).WillByDefault(Return(-4.0));
    InSequence order;
    EXPECT_CALL(start_path, start_path(-4.0));
    EXPECT_CALL(add_step, add_step(kRoot, 1));
    EXPECT_CALL(add_step, add_step(5, 0));
    EXPECT_CALL(add_step, add_step(8, 1));
    EXPECT_CALL(clear_choices, clear_choices());

    sut.record();
}

TEST_F(BestPathRecorderTest, AWorseEpisodeLeavesTheBestPathAlone)
{
    ON_CALL(get_reward, value()).WillByDefault(Return(-12.0));
    EXPECT_CALL(start_path, start_path(_)).Times(0);
    EXPECT_CALL(add_step, add_step(_, _)).Times(0);
    EXPECT_CALL(clear_choices, clear_choices());

    sut.record();
}

TEST_F(BestPathRecorderTest, AnEqualEpisodeLeavesTheBestPathAlone)
{
    ON_CALL(get_reward, value()).WillByDefault(Return(-10.0));
    EXPECT_CALL(start_path, start_path(_)).Times(0);

    sut.record();
}

TEST_F(BestPathRecorderTest, AnEpisodeWithoutChoicesRecordsAnEmptyPath)
{
    std::vector<int> none;
    ON_CALL(get_choices, choices()).WillByDefault(ReturnRef(none));
    ON_CALL(get_reward, value()).WillByDefault(Return(-1.0));
    EXPECT_CALL(start_path, start_path(-1.0));
    EXPECT_CALL(add_step, add_step(_, _)).Times(0);

    sut.record();
}
