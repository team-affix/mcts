// best_path_recording_terminator terminates as the wrapped terminator does,
// then records the episode's path.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "infrastructure/best_path_recording_terminator.hpp"

using ::testing::InSequence;
using ::testing::StrictMock;

namespace
{

struct MockTerminate
{
    MOCK_METHOD(void, terminate, (), ());
};

struct MockRecord
{
    MOCK_METHOD(void, record, (), ());
};

}

class BestPathRecordingTerminatorTest : public ::testing::Test
{
protected:
    StrictMock<MockTerminate> terminator;
    StrictMock<MockRecord>    recorder;
    monte_carlo::best_path_recording_terminator<MockTerminate, MockRecord> sut{terminator, recorder};
};

TEST_F(BestPathRecordingTerminatorTest, TerminatesBeforeRecording)
{
    InSequence order;
    EXPECT_CALL(terminator, terminate());
    EXPECT_CALL(recorder, record());

    sut.terminate();
}

TEST_F(BestPathRecordingTerminatorTest, EachTerminationTerminatesAndRecordsOnce)
{
    InSequence order;
    EXPECT_CALL(terminator, terminate());
    EXPECT_CALL(recorder, record());
    EXPECT_CALL(terminator, terminate());
    EXPECT_CALL(recorder, record());

    sut.terminate();
    sut.terminate();
}
