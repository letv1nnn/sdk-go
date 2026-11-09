// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1

import (
	"context"
	"sync"
	"testing"
	"time"

	dm "github.com/ai-openshell/sdk-go/proto/datamodelv1"
	pb "github.com/ai-openshell/sdk-go/proto/openshellv1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// watchLogsMock returns a mock server with one Ready sandbox registered so
// WatchLogs' name resolution succeeds.
func watchLogsMock() *mockSandboxServer {
	mock := newMockSandboxServer()
	mock.sandboxes["sb-1"] = &pb.Sandbox{
		Metadata: &dm.ObjectMeta{Id: "id-1", Name: "sb-1"},
		Status:   &pb.SandboxStatus{Phase: pb.SandboxPhase_SANDBOX_PHASE_READY},
	}
	return mock
}

func logEvent(cursor, message string) *pb.SandboxStreamEvent {
	return &pb.SandboxStreamEvent{
		Cursor:  cursor,
		Payload: &pb.SandboxStreamEvent_Log{Log: &pb.SandboxLogLine{Message: message, Level: "INFO"}},
	}
}

func platformEvent(cursor, reason string) *pb.SandboxStreamEvent {
	return &pb.SandboxStreamEvent{
		Cursor: cursor,
		Payload: &pb.SandboxStreamEvent_Event{Event: &pb.PlatformEvent{
			Reason: reason,
			Type:   "Normal",
			Source: "kubernetes",
		}},
	}
}

func warningEvent(message string) *pb.SandboxStreamEvent {
	return &pb.SandboxStreamEvent{
		Payload: &pb.SandboxStreamEvent_Warning{Warning: &pb.SandboxStreamWarning{Message: message}},
	}
}

// observeWatchLogsBackoff records every reconnect backoff the client requests.
// With wait false the delay is skipped, so a test can drive many retries
// instantly; with wait true the watcher still performs its real, cancellable
// wait. The returned func snapshots the recorded delays.
func observeWatchLogsBackoff(c *sandboxClient, wait bool) func() []time.Duration {
	var mu sync.Mutex
	var delays []time.Duration
	c.watchLogsWait = func(ctx context.Context, w *watcher[*WatchLogEvent], d time.Duration) bool {
		mu.Lock()
		delays = append(delays, d)
		mu.Unlock()
		if wait {
			return w.sleep(ctx, d)
		}
		return !w.stopped() && ctx.Err() == nil
	}
	return func() []time.Duration {
		mu.Lock()
		defer mu.Unlock()
		return append([]time.Duration(nil), delays...)
	}
}

// drainWatchLogs collects every event until the channel closes.
func drainWatchLogs(t *testing.T, w WatchInterface[*WatchLogEvent]) []Event[*WatchLogEvent] {
	t.Helper()
	var events []Event[*WatchLogEvent]
	timeout := time.After(10 * time.Second)
	for {
		select {
		case ev, ok := <-w.ResultChan():
			if !ok {
				return events
			}
			events = append(events, ev)
		case <-timeout:
			t.Fatal("timed out waiting for WatchLogs channel to close")
		}
	}
}

func TestWatchLogs_DeliversLogsEventsAndWarnings(t *testing.T) {
	mock := watchLogsMock()
	mock.watchEvents = []*pb.SandboxStreamEvent{
		{
			Cursor: "0000000001",
			Payload: &pb.SandboxStreamEvent_Log{Log: &pb.SandboxLogLine{
				Message:   "hello",
				Level:     "INFO",
				Target:    "supervisor",
				Source:    "sandbox",
				EventTime: timestamppb.New(time.UnixMilli(1700000000000)),
				Fields:    map[string]string{"dst_host": "example.test"},
			}},
		},
		platformEvent("0000000002", "Started"),
		warningEvent("dropped 3 messages"),
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1", WatchLogsOptions{
		FollowLogs:   true,
		FollowEvents: true,
	})
	require.NoError(t, err)
	defer w.Stop()

	events := drainWatchLogs(t, w)
	require.Len(t, events, 3)

	require.Equal(t, EventAdded, events[0].Type)
	logItem := events[0].Object
	require.NotNil(t, logItem)
	assert.Equal(t, WatchLogKindLog, logItem.Kind)
	assert.Equal(t, "0000000001", logItem.Cursor)
	require.NotNil(t, logItem.Log)
	assert.Equal(t, "hello", logItem.Log.Message)
	assert.Equal(t, "sandbox", logItem.Log.Source)
	assert.Equal(t, map[string]string{"dst_host": "example.test"}, logItem.Log.Fields)
	assert.Nil(t, logItem.Event)

	platformItem := events[1].Object
	require.NotNil(t, platformItem)
	assert.Equal(t, WatchLogKindEvent, platformItem.Kind)
	assert.Equal(t, "0000000002", platformItem.Cursor)
	require.NotNil(t, platformItem.Event)
	assert.Equal(t, "Started", platformItem.Event.Reason)
	assert.Nil(t, platformItem.Log)

	warnItem := events[2].Object
	require.NotNil(t, warnItem)
	assert.Equal(t, WatchLogKindWarning, warnItem.Kind)
	assert.Equal(t, "dropped 3 messages", warnItem.Warning)
	// Warnings are not resumable, so they carry no cursor.
	assert.Empty(t, warnItem.Cursor)
}

func TestWatchLogs_SkipsNonResumablePayloads(t *testing.T) {
	mock := watchLogsMock()
	mock.watchEvents = []*pb.SandboxStreamEvent{
		{Payload: &pb.SandboxStreamEvent_Sandbox{Sandbox: &pb.Sandbox{
			Metadata: &dm.ObjectMeta{Name: "sb-1"},
			Status:   &pb.SandboxStatus{Phase: pb.SandboxPhase_SANDBOX_PHASE_READY},
		}}},
		logEvent("0000000001", "only this"),
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()

	events := drainWatchLogs(t, w)
	require.Len(t, events, 1)
	assert.Equal(t, WatchLogKindLog, events[0].Object.Kind)
	assert.Equal(t, "only this", events[0].Object.Log.Message)
}

func TestWatchLogs_RequestFields(t *testing.T) {
	mock := watchLogsMock()
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1", WatchLogsOptions{
		FollowLogs:        true,
		FollowEvents:      true,
		LogSources:        []string{"sandbox"},
		LogMinLevel:       "WARN",
		ResumeAfterCursor: "0000000007",
		LogTailLines:      25,
		EventTail:         5,
	})
	require.NoError(t, err)
	defer w.Stop()

	drainWatchLogs(t, w)

	mock.mu.Lock()
	req := mock.watchRequest
	mock.mu.Unlock()
	require.NotNil(t, req)
	assert.Equal(t, "sb-1", req.GetSandbox())
	assert.Equal(t, "default", req.GetWorkspaceScope().GetWorkspace())
	assert.False(t, req.GetFollowStatus())
	assert.True(t, req.GetFollowLogs())
	assert.True(t, req.GetFollowEvents())
	assert.Equal(t, []string{"sandbox"}, req.GetLogSources())
	assert.Equal(t, "WARN", req.GetLogMinLevel())
	assert.Equal(t, "0000000007", req.GetResumeAfterCursor())
	assert.Equal(t, uint32(25), req.GetLogTailLines())
	assert.Equal(t, uint32(5), req.GetEventTail())
}

func TestWatchLogs_DefaultsToFollowingLogs(t *testing.T) {
	mock := watchLogsMock()
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()

	drainWatchLogs(t, w)

	mock.mu.Lock()
	req := mock.watchRequest
	mock.mu.Unlock()
	require.NotNil(t, req)
	assert.True(t, req.GetFollowLogs())
	assert.False(t, req.GetFollowEvents())
}

func TestWatchLogs_ReconnectResumesFromHighestCursor(t *testing.T) {
	mock := watchLogsMock()
	mock.watchFunc = func(attempt int, _ *pb.WatchSandboxRequest, stream grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		switch attempt {
		case 0:
			if err := stream.Send(logEvent("0000000001", "first")); err != nil {
				return err
			}
			if err := stream.Send(platformEvent("0000000003", "Pulled")); err != nil {
				return err
			}
			return status.Error(codes.Unavailable, "connection reset")
		default:
			return stream.Send(logEvent("0000000004", "after resume"))
		}
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1", WatchLogsOptions{
		FollowLogs:   true,
		FollowEvents: true,
		LogSources:   []string{"sandbox", "gateway"},
		LogMinLevel:  "INFO",
		LogTailLines: 10,
		EventTail:    3,
	})
	require.NoError(t, err)
	defer w.Stop()

	events := drainWatchLogs(t, w)
	require.Len(t, events, 3, "reconnect must not drop or duplicate events")
	assert.Equal(t, "first", events[0].Object.Log.Message)
	assert.Equal(t, "Pulled", events[1].Object.Event.Reason)
	assert.Equal(t, "after resume", events[2].Object.Log.Message)
	for _, ev := range events {
		assert.NotEqual(t, EventError, ev.Type)
	}

	mock.mu.Lock()
	requests := mock.watchRequests
	mock.mu.Unlock()
	require.Len(t, requests, 2)
	assert.Empty(t, requests[0].GetResumeAfterCursor())
	assert.Equal(t, "0000000003", requests[1].GetResumeAfterCursor())

	// The redial must rebuild the same request, changing only the cursor.
	want := proto.Clone(requests[0]).(*pb.WatchSandboxRequest)
	want.ResumeAfterCursor = "0000000003"
	assert.True(t, proto.Equal(want, requests[1]), "redial request differs beyond the cursor:\nwant %v\ngot  %v", want, requests[1])
}

func TestWatchLogs_CursorIsHighWaterMark(t *testing.T) {
	// The gateway delivers events in cursor order, but the SDK must still never
	// rewind its resume point if a lower cursor arrives after a higher one, as
	// the proto contract says to keep the greater cursor.
	mock := watchLogsMock()
	mock.watchFunc = func(attempt int, _ *pb.WatchSandboxRequest, stream grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		if attempt == 0 {
			if err := stream.Send(logEvent("0000000005", "high")); err != nil {
				return err
			}
			if err := stream.Send(platformEvent("0000000002", "Late")); err != nil {
				return err
			}
			return status.Error(codes.Unavailable, "connection reset")
		}
		return nil
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1", WatchLogsOptions{
		FollowLogs:   true,
		FollowEvents: true,
	})
	require.NoError(t, err)
	defer w.Stop()

	drainWatchLogs(t, w)

	mock.mu.Lock()
	requests := mock.watchRequests
	mock.mu.Unlock()
	require.Len(t, requests, 2)
	assert.Equal(t, "0000000005", requests[1].GetResumeAfterCursor())
}

func TestWatchLogs_WarningDoesNotAdvanceCursor(t *testing.T) {
	mock := watchLogsMock()
	mock.watchFunc = func(attempt int, _ *pb.WatchSandboxRequest, stream grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		if attempt == 0 {
			if err := stream.Send(logEvent("0000000001", "before")); err != nil {
				return err
			}
			if err := stream.Send(warningEvent("dropped 2 messages")); err != nil {
				return err
			}
			return status.Error(codes.Unavailable, "connection reset")
		}
		return nil
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()

	events := drainWatchLogs(t, w)
	require.Len(t, events, 2)
	assert.Equal(t, WatchLogKindWarning, events[1].Object.Kind)

	mock.mu.Lock()
	requests := mock.watchRequests
	mock.mu.Unlock()
	require.Len(t, requests, 2)
	assert.Equal(t, "0000000001", requests[1].GetResumeAfterCursor())
}

func TestWatchLogs_OutOfRangeIsTerminal(t *testing.T) {
	mock := watchLogsMock()
	mock.watchFunc = func(_ int, _ *pb.WatchSandboxRequest, stream grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		if err := stream.Send(logEvent("0000000001", "before the gap")); err != nil {
			return err
		}
		return status.Error(codes.OutOfRange, "resume cursor was trimmed")
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()

	events := drainWatchLogs(t, w)
	require.Len(t, events, 2)
	assert.Equal(t, WatchLogKindLog, events[0].Object.Kind)

	last := events[1]
	assert.Equal(t, EventError, last.Type)
	require.Error(t, last.Err)
	assert.True(t, IsOutOfRange(last.Err), "want ErrorOutOfRange, got %v", last.Err)
	assert.False(t, IsInvalidArgument(last.Err))

	mock.mu.Lock()
	attempts := len(mock.watchRequests)
	mock.mu.Unlock()
	assert.Equal(t, 1, attempts, "OUT_OF_RANGE must not be retried")
}

func TestWatchLogs_NonRetryableErrorIsTerminal(t *testing.T) {
	// ResourceExhausted maps to ErrorUnavailable in the SDK, but it must not be
	// redialed: the retry decision uses the raw gRPC code.
	for _, code := range []codes.Code{
		codes.PermissionDenied,
		codes.Unauthenticated,
		codes.ResourceExhausted,
		codes.Internal,
	} {
		t.Run(code.String(), func(t *testing.T) {
			mock := watchLogsMock()
			mock.watchFunc = func(_ int, _ *pb.WatchSandboxRequest, _ grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
				return status.Error(code, "nope")
			}
			client, cleanup := setupSandboxTest(t, mock)
			defer cleanup()

			w, err := client.WatchLogs(context.Background(), "default", "sb-1")
			require.NoError(t, err)
			defer w.Stop()

			events := drainWatchLogs(t, w)
			require.Len(t, events, 1)
			assert.Equal(t, EventError, events[0].Type)
			var se *StatusError
			require.ErrorAs(t, events[0].Err, &se)
			assert.Equal(t, int32(code), se.GRPCCode)

			mock.mu.Lock()
			attempts := len(mock.watchRequests)
			mock.mu.Unlock()
			assert.Equal(t, 1, attempts, "%s must not be retried", code)
		})
	}
}

func TestWatchLogs_BackoffResetsAfterDeliveredEvent(t *testing.T) {
	// Two drops separated by a delivered event must both request the initial
	// backoff, not an escalating one.
	const attempts = 3
	mock := watchLogsMock()
	mock.watchFunc = func(attempt int, _ *pb.WatchSandboxRequest, stream grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		if attempt >= attempts-1 {
			return nil
		}
		if err := stream.Send(logEvent("000000000"+string(rune('1'+attempt)), "tick")); err != nil {
			return err
		}
		return status.Error(codes.Unavailable, "connection reset")
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()
	backoffs := observeWatchLogsBackoff(client, false)

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()

	events := drainWatchLogs(t, w)
	require.Len(t, events, attempts-1)

	// Both backoff requests should be at the initial delay (reset between drops).
	assert.Equal(t, []time.Duration{watchLogsInitialBackoff, watchLogsInitialBackoff}, backoffs())
}

func TestWatchLogs_BackoffGrowsWhenOnlySnapshotsArrive(t *testing.T) {
	// The gateway opens every stream with a status snapshot. A gateway that
	// sends it and then drops has delivered nothing, so the backoff must keep
	// growing instead of resetting on each attempt.
	const attempts = 4
	mock := watchLogsMock()

	mock.watchFunc = func(attempt int, _ *pb.WatchSandboxRequest, stream grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		if attempt >= attempts-1 {
			return nil
		}
		snapshot := &pb.SandboxStreamEvent{Payload: &pb.SandboxStreamEvent_Sandbox{Sandbox: &pb.Sandbox{
			Metadata: &dm.ObjectMeta{Name: "sb-1"},
			Status:   &pb.SandboxStatus{Phase: pb.SandboxPhase_SANDBOX_PHASE_READY},
		}}}
		if err := stream.Send(snapshot); err != nil {
			return err
		}
		return status.Error(codes.Unavailable, "connection reset")
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()
	backoffs := observeWatchLogsBackoff(client, false)

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()

	assert.Empty(t, drainWatchLogs(t, w))

	assert.Equal(t, []time.Duration{
		watchLogsInitialBackoff,
		2 * watchLogsInitialBackoff,
		4 * watchLogsInitialBackoff,
	}, backoffs())
}

func TestWatchLogs_BackoffCapsAtMax(t *testing.T) {
	// A gateway that stays unavailable must be redialed at the capped delay,
	// not at an ever-growing one.
	const retries = 8
	mock := watchLogsMock()
	mock.watchFunc = func(attempt int, _ *pb.WatchSandboxRequest, _ grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		if attempt >= retries {
			return nil
		}
		return status.Error(codes.Unavailable, "gateway down")
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()
	backoffs := observeWatchLogsBackoff(client, false)

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()

	assert.Empty(t, drainWatchLogs(t, w))

	assert.Equal(t, []time.Duration{
		100 * time.Millisecond,
		200 * time.Millisecond,
		400 * time.Millisecond,
		800 * time.Millisecond,
		1600 * time.Millisecond,
		watchLogsMaxBackoff,
		watchLogsMaxBackoff,
		watchLogsMaxBackoff,
	}, backoffs())
}

func TestWatchLogs_BackoffWaitsOncePerRetry(t *testing.T) {
	// Each retry must wait the requested backoff once. Three retries from the
	// initial delay add up to 100+200+400ms; waiting each delay twice would
	// double that.
	mock := watchLogsMock()
	var mu sync.Mutex
	var dials []time.Time
	mock.watchFunc = func(attempt int, _ *pb.WatchSandboxRequest, _ grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		mu.Lock()
		dials = append(dials, time.Now())
		mu.Unlock()
		if attempt >= 3 {
			return nil
		}
		return status.Error(codes.Unavailable, "connection reset")
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()
	drainWatchLogs(t, w)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, dials, 4)
	const expected = 7 * watchLogsInitialBackoff
	elapsed := dials[3].Sub(dials[0])
	assert.GreaterOrEqual(t, elapsed, expected)
	assert.Less(t, elapsed, expected+expected/2, "retries waited the backoff more than once")
}

func TestWatchLogs_StopInterruptsBackoff(t *testing.T) {
	// Stop during a reconnect backoff must return promptly instead of waiting
	// out the delay, which grows to the 2s cap against a gateway that stays
	// unavailable.
	const longBackoff = 8 * watchLogsInitialBackoff
	mock := watchLogsMock()
	mock.watchFunc = func(int, *pb.WatchSandboxRequest, grpc.ServerStreamingServer[pb.SandboxStreamEvent]) error {
		return status.Error(codes.Unavailable, "gateway down")
	}
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	// Skip the short delays, then perform the first long one for real so Stop
	// has a backoff to interrupt.
	inLongBackoff := make(chan struct{})
	var once sync.Once
	client.watchLogsWait = func(ctx context.Context, w *watcher[*WatchLogEvent], d time.Duration) bool {
		if d < longBackoff {
			return !w.stopped() && ctx.Err() == nil
		}
		once.Do(func() { close(inLongBackoff) })
		return w.sleep(ctx, d)
	}

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)

	select {
	case <-inLongBackoff:
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for a long backoff")
	}

	start := time.Now()
	w.Stop()
	select {
	case _, ok := <-w.ResultChan():
		assert.False(t, ok, "channel should close after Stop")
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for channel close after Stop")
	}
	assert.Less(t, time.Since(start), longBackoff/2, "Stop waited out the backoff")
}

func TestWatchLogs_StopClosesChannel(t *testing.T) {
	mock := watchLogsMock()
	keepOpen := make(chan struct{})
	defer close(keepOpen)
	mock.watchEvents = []*pb.SandboxStreamEvent{logEvent("0000000001", "hello")}
	mock.watchKeepOpen = keepOpen
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	w, err := client.WatchLogs(context.Background(), "default", "sb-1")
	require.NoError(t, err)

	ev := <-w.ResultChan()
	assert.Equal(t, "hello", ev.Object.Log.Message)

	w.Stop()
	// Nothing further is queued, and Stop must not surface the cancelled
	// stream as an ERROR event.
	for _, ev := range drainWatchLogs(t, w) {
		assert.NotEqual(t, EventError, ev.Type, "Stop must not emit an error: %v", ev.Err)
	}
}

func TestWatchLogs_ContextCancelClosesChannel(t *testing.T) {
	mock := watchLogsMock()
	keepOpen := make(chan struct{})
	defer close(keepOpen)
	mock.watchKeepOpen = keepOpen
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	w, err := client.WatchLogs(ctx, "default", "sb-1")
	require.NoError(t, err)
	defer w.Stop()

	cancel()
	// The mock sends nothing, and cancellation must not surface the
	// cancelled stream as an ERROR event.
	assert.Empty(t, drainWatchLogs(t, w))
}

func TestWatchLogs_EmptyName(t *testing.T) {
	mock := watchLogsMock()
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	_, err := client.WatchLogs(context.Background(), "default", "")
	require.Error(t, err)
	assert.True(t, IsInvalidArgument(err))
}

func TestWatchLogs_UnknownSandbox(t *testing.T) {
	mock := watchLogsMock()
	client, cleanup := setupSandboxTest(t, mock)
	defer cleanup()

	_, err := client.WatchLogs(context.Background(), "default", "missing")
	require.Error(t, err)
	assert.True(t, IsNotFound(err))

	mock.mu.Lock()
	attempts := len(mock.watchRequests)
	mock.mu.Unlock()
	assert.Zero(t, attempts, "resolution failure must not open a stream")
}
