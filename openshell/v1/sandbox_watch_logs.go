// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1

import (
	"context"
	"errors"
	"io"
	"time"

	"github.com/ai-openshell/sdk-go/openshell/v1/internal/converter"
	"github.com/ai-openshell/sdk-go/openshell/v1/types"
	pb "github.com/ai-openshell/sdk-go/proto/openshellv1"
	"google.golang.org/grpc/codes"
)

const (
	// watchLogsInitialBackoff is the delay before the first reconnect attempt.
	// It is restored after every delivered log, event, or warning, so a
	// long-lived stream that drops retries promptly instead of at the capped
	// delay.
	watchLogsInitialBackoff = 100 * time.Millisecond
	// watchLogsMaxBackoff caps the reconnect delay.
	watchLogsMaxBackoff = 2 * time.Second
	// watchLogsChannelBuffer is the result channel depth.
	watchLogsChannelBuffer = 64
)

// WatchLogs streams a sandbox's log lines and platform events with loss-aware
// resume.
//
// Each delivered event carries an opaque cursor. The watcher tracks the highest
// cursor it has seen, reconnects transparently on transient stream errors, and
// asks the gateway to replay only what follows that cursor, so a reconnect
// neither loses nor duplicates events. This relies on the gateway delivering
// live log and platform events in cursor order across both sources, which it
// guarantees by merging them and withholding any event above its publication
// watermark.
//
// Only gRPC Unavailable (a dropped connection or gateway restart) is retried,
// with capped exponential backoff, and it is retried indefinitely: no event is
// emitted while the watcher reconnects, so a gateway that stays down looks
// like a quiet sandbox. Bound the watch with a context deadline or call Stop
// when that matters. This differs from Watch, which reports Unavailable as an
// ERROR event immediately. A connection that dies without the transport
// noticing (no keepalive, half-open TCP) stalls until the context ends.
//
// Loss is never silent. A recoverable server-side gap arrives as a
// WatchLogKindWarning event and the stream continues. An unrecoverable gap ends
// the stream with an ERROR event carrying ErrorOutOfRange: the resume point is
// gone, so the events after it cannot be replayed. That error is deliberately
// not retried, because restarting from the tail would hide the loss this API
// exists to surface. A caller that accepts the gap can start a new watch with
// an empty ResumeAfterCursor; retrying the same cursor fails identically.
//
// The returned watcher's channel closes when the server ends the stream, when
// the context is cancelled, or after Stop. Call Stop to release the stream.
func (s *sandboxClient) WatchLogs(ctx context.Context, workspace, name string, opts ...WatchLogsOptions) (WatchInterface[*WatchLogEvent], error) {
	if name == "" {
		return nil, &StatusError{Code: ErrorInvalidArgument, Message: "sandbox name must not be empty"}
	}

	var cfg WatchLogsOptions
	if len(opts) > 0 {
		cfg = opts[0]
	}
	if !cfg.FollowLogs && !cfg.FollowEvents {
		cfg.FollowLogs = true
	}

	// Resolve and confirm the sandbox under this workspace before opening the
	// stream, so a bad name fails here rather than inside the reconnect loop.
	sb, err := s.Get(ctx, workspace, name)
	if err != nil {
		return nil, err
	}
	resolved := name
	if sb != nil && sb.Name != "" {
		resolved = sb.Name
	}

	streamCtx, streamCancel := context.WithCancel(ctx)
	ch := make(chan Event[*WatchLogEvent], watchLogsChannelBuffer)
	w := newWatcher(ch, streamCancel)

	go s.runWatchLogs(streamCtx, w, workspace, resolved, cfg)

	return w, nil
}

// runWatchLogs drives the reconnect/resume loop until the stream ends cleanly,
// a terminal error occurs, or the consumer stops watching.
func (s *sandboxClient) runWatchLogs(ctx context.Context, w *watcher[*WatchLogEvent], workspace, name string, cfg WatchLogsOptions) {
	defer close(w.result)
	defer w.cancel()

	cursor := cfg.ResumeAfterCursor
	backoff := watchLogsInitialBackoff

	for {
		req := &pb.WatchSandboxRequest{
			Sandbox:           name,
			WorkspaceScope:    namedWorkspaceScope(workspace),
			FollowStatus:      false,
			FollowLogs:        cfg.FollowLogs,
			FollowEvents:      cfg.FollowEvents,
			LogTailLines:      cfg.LogTailLines,
			EventTail:         cfg.EventTail,
			LogSources:        converter.CopyStringSlice(cfg.LogSources),
			LogMinLevel:       cfg.LogMinLevel,
			ResumeAfterCursor: cursor,
		}

		stream, err := s.client.WatchSandbox(ctx, req)
		if err != nil {
			if !s.retryWatchLogs(ctx, w, err, &backoff) {
				return
			}
			continue
		}

		reconnect := false
		for {
			ev, recvErr := stream.Recv()
			if recvErr != nil {
				if errors.Is(recvErr, io.EOF) {
					// Server closed the stream cleanly; nothing left to resume.
					return
				}
				reconnect = s.retryWatchLogs(ctx, w, recvErr, &backoff)
				break
			}

			item := converter.WatchLogEventFromProto(ev)
			if item == nil {
				continue
			}
			// A delivered item means the connection is healthy again. The
			// status snapshot the gateway sends at the start of every stream
			// is skipped above and must not count: a gateway that accepts the
			// stream and then drops it would otherwise be redialed at the
			// initial delay forever.
			backoff = watchLogsInitialBackoff
			// The gateway delivers log and platform events in cursor order
			// across both sources: it merges what each source has queued and
			// withholds anything above its publication watermark, so a
			// higher cursor is never sent ahead of a lower one still in
			// flight. Comparing instead of assigning is defensive: the proto
			// contract is to keep the greater cursor, and a rewind would
			// replay already-delivered events after a reconnect.
			if item.Cursor > cursor {
				cursor = item.Cursor
			}
			if !w.emit(ctx, Event[*WatchLogEvent]{Type: EventAdded, Object: item}) {
				return
			}
		}
		if !reconnect {
			return
		}
	}
}

// retryWatchLogs decides what to do with a stream error. It returns true when
// the caller should redial from the tracked cursor, having already waited out
// the backoff and doubled it. Otherwise the error is terminal: it is reported
// as an ERROR event (unless the consumer already stopped) and the watch ends.
func (s *sandboxClient) retryWatchLogs(ctx context.Context, w *watcher[*WatchLogEvent], err error, backoff *time.Duration) bool {
	if w.stopped() || ctx.Err() != nil {
		return false
	}
	converted := converter.FromGRPCError(err)
	if !isRetryableWatchLogsError(converted) {
		// Terminal, including ErrorOutOfRange: the resume point is gone, and
		// restarting from the tail here would silently swallow the gap.
		w.emit(ctx, Event[*WatchLogEvent]{Type: EventError, Err: converted})
		return false
	}
	wait := w.sleep
	if s.watchLogsWait != nil {
		wait = func(ctx context.Context, d time.Duration) bool { return s.watchLogsWait(ctx, w, d) }
	}
	if !wait(ctx, *backoff) {
		return false
	}
	*backoff = min(*backoff*2, watchLogsMaxBackoff)
	return true
}

// isRetryableWatchLogsError reports whether a converted stream error is a
// transient condition worth reconnecting on. Only gRPC Unavailable (connection
// drop, gateway restart) qualifies. It checks the raw gRPC code, not the SDK
// ErrorCode, because ResourceExhausted also maps to ErrorUnavailable and must
// not be redialed. Every other status is surfaced to the caller instead of
// looping.
func isRetryableWatchLogsError(err error) bool {
	var se *types.StatusError
	if errors.As(err, &se) {
		return se.GRPCCode == int32(codes.Unavailable)
	}
	return false
}
