package openai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// recordingSSECallback records every candidate offered by the parser.
type recordingSSECallback struct {
	calls []string
	fn    func(data string) error
}

func (r *recordingSSECallback) onData(data string) error {
	r.calls = append(r.calls, data)
	if r.fn != nil {
		return r.fn(data)
	}
	return nil
}

func decodeFailure(data string) error {
	return fmt.Errorf("fixture stream: decode error: %q", len(data))
}

func requireSSELimitError(t *testing.T, err error, reason string) *sseResourceLimitError {
	t.Helper()
	var limitErr *sseResourceLimitError
	if !errors.As(err, &limitErr) {
		t.Fatalf("err = %v, want sseResourceLimitError(%s)", err, reason)
	}
	if limitErr.Reason != reason {
		t.Fatalf("limit reason = %q, want %q", limitErr.Reason, reason)
	}
	return limitErr
}

func TestIssue152AggregateByteBoundary(t *testing.T) {
	const budget = 16
	for _, size := range []int{budget - 1, budget, budget + 1} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			payload := strings.Repeat("x", size)
			rec := &recordingSSECallback{}
			err := consumeSSEWithLimits(strings.NewReader("data: "+payload+"\n\n"), rec.onData, sseLimits{maxEventBytes: budget, maxDecodeAttempts: 2})
			if size <= budget {
				if err != nil {
					t.Fatalf("size %d: err = %v", size, err)
				}
				if len(rec.calls) != 1 || rec.calls[0] != payload {
					t.Fatalf("size %d: calls = %q", size, rec.calls)
				}
				return
			}
			requireSSELimitError(t, err, sseLimitEventBytes)
			if len(rec.calls) != 0 {
				t.Fatalf("over-budget candidate reached callback: %d calls", len(rec.calls))
			}
		})
	}
}

func TestIssue152MultiLineAggregateCountsJoinSeparators(t *testing.T) {
	input := "data: abc\ndata: abc\ndata: abc\n\n"
	rec := &recordingSSECallback{}
	if err := consumeSSEWithLimits(strings.NewReader(input), rec.onData, sseLimits{maxEventBytes: 11, maxDecodeAttempts: 2}); err != nil {
		t.Fatalf("budget 11: %v", err)
	}
	if len(rec.calls) != 1 || rec.calls[0] != "abc\nabc\nabc" {
		t.Fatalf("calls = %q, want one joined 11-byte candidate", rec.calls)
	}

	rec = &recordingSSECallback{}
	err := consumeSSEWithLimits(strings.NewReader(input), rec.onData, sseLimits{maxEventBytes: 10, maxDecodeAttempts: 2})
	requireSSELimitError(t, err, sseLimitEventBytes)
	if len(rec.calls) != 0 {
		t.Fatalf("callback ran %d times for over-budget aggregate", len(rec.calls))
	}
}

// endlessDataReader returns short data lines without ever sending a blank line
// or EOF. After maxReads it returns a sentinel so a parser that ignores the
// aggregate budget fails the test instead of looping forever.
type endlessDataReader struct {
	line     []byte
	reads    int
	maxReads int
	bytes    int
}

var errEndlessReaderExhausted = errors.New("endless reader exhausted")

func (r *endlessDataReader) Read(p []byte) (int, error) {
	if r.reads >= r.maxReads {
		return 0, errEndlessReaderExhausted
	}
	r.reads++
	n := copy(p, r.line)
	r.bytes += n
	return n, nil
}

func TestIssue152ShortDataLinesStopAtBudgetBeforeEOF(t *testing.T) {
	const budget = 64
	reader := &endlessDataReader{line: []byte("data: x\n"), maxReads: 10_000}
	rec := &recordingSSECallback{}
	err := consumeSSEWithLimits(reader, rec.onData, sseLimits{maxEventBytes: budget, maxDecodeAttempts: 2})
	requireSSELimitError(t, err, sseLimitEventBytes)
	if len(rec.calls) != 0 {
		t.Fatalf("callback ran before the unterminated event was complete: %d", len(rec.calls))
	}
	// Each "x" line retains two bytes (payload plus join separator), so the
	// budget is exhausted after about budget/2 lines; allow scanner read-ahead
	// of one line per Read call but never an unbounded scan.
	if reader.reads > budget {
		t.Fatalf("parser read %d lines (%d bytes) for a %d-byte budget", reader.reads, reader.bytes, budget)
	}
}

func TestIssue152EmptyLinesUnicodeAndCRLFAreMetered(t *testing.T) {
	// Empty data lines contribute only separators, which still count.
	rec := &recordingSSECallback{}
	empty := strings.Repeat("data:\r\n", 20) + "\r\n"
	err := consumeSSEWithLimits(strings.NewReader(empty), rec.onData, sseLimits{maxEventBytes: 8, maxDecodeAttempts: 2})
	requireSSELimitError(t, err, sseLimitEventBytes)
	if len(rec.calls) != 0 {
		t.Fatalf("empty-line aggregate reached callback: %d", len(rec.calls))
	}

	// Unicode is metered in UTF-8 bytes: three runes are nine bytes.
	han := "中文字"
	rec = &recordingSSECallback{}
	if err := consumeSSEWithLimits(strings.NewReader("data: "+han+"\r\n\r\n"), rec.onData, sseLimits{maxEventBytes: 9, maxDecodeAttempts: 2}); err != nil {
		t.Fatalf("nine-byte unicode payload: %v", err)
	}
	if len(rec.calls) != 1 || rec.calls[0] != han {
		t.Fatalf("calls = %q", rec.calls)
	}
	rec = &recordingSSECallback{}
	err = consumeSSEWithLimits(strings.NewReader("data: "+han+"\r\n\r\n"), rec.onData, sseLimits{maxEventBytes: 8, maxDecodeAttempts: 2})
	requireSSELimitError(t, err, sseLimitEventBytes)
}

func TestIssue152PendingPlusFragmentSharesBudget(t *testing.T) {
	// First group fails decode (10 bytes retained). Next fragment adds a join
	// separator plus 6 bytes = 17 > 16, so it must be rejected before retry.
	input := "data: 0123456789\n\ndata: abcdef\n\n"
	rec := &recordingSSECallback{fn: decodeFailure}
	err := consumeSSEWithLimits(strings.NewReader(input), rec.onData, sseLimits{maxEventBytes: 16, maxDecodeAttempts: 5})
	requireSSELimitError(t, err, sseLimitEventBytes)
	if len(rec.calls) != 1 {
		t.Fatalf("callback calls = %d, want only the first failed attempt", len(rec.calls))
	}

	// The same bytes fit when the budget covers pending + separator + fragment.
	rec = &recordingSSECallback{fn: func(data string) error {
		if data == "0123456789\nabcdef" {
			return nil
		}
		return decodeFailure(data)
	}}
	if err := consumeSSEWithLimits(strings.NewReader(input), rec.onData, sseLimits{maxEventBytes: 17, maxDecodeAttempts: 5}); err != nil {
		t.Fatalf("exact-budget reassembly: %v", err)
	}
	if len(rec.calls) != 2 {
		t.Fatalf("calls = %q", rec.calls)
	}
}

func TestIssue152DecodeAttemptsResetOnlyAfterSuccess(t *testing.T) {
	const attempts = 3
	// Two failures, success on the third attempt, then an independent event that
	// again needs two failures before succeeding: both groups fit the budget.
	input := "data: {\n\ndata: \"a\":\n\ndata: 1}\n\n" +
		"data: {\n\ndata: \"b\":\n\ndata: 2}\n\n"
	successes := 0
	rec := &recordingSSECallback{fn: func(data string) error {
		if strings.HasSuffix(data, "}") {
			successes++
			return nil
		}
		return decodeFailure(data)
	}}
	if err := consumeSSEWithLimits(strings.NewReader(input), rec.onData, sseLimits{maxEventBytes: 64, maxDecodeAttempts: attempts}); err != nil {
		t.Fatalf("K-1 failures then success: %v", err)
	}
	if successes != 2 || len(rec.calls) != 6 {
		t.Fatalf("successes=%d calls=%q", successes, rec.calls)
	}
	if rec.calls[5] != "{\n\"b\":\n2}" {
		t.Fatalf("second event reused prior pending data: %q", rec.calls[5])
	}
}

func TestIssue152DecodeAttemptLimitStopsWithoutWaitingForEOF(t *testing.T) {
	const attempts = 3
	reader := &endlessDataReader{line: []byte("data: {\n\n"), maxReads: 1_000}
	rec := &recordingSSECallback{fn: decodeFailure}
	err := consumeSSEWithLimits(reader, rec.onData, sseLimits{maxEventBytes: 1 << 20, maxDecodeAttempts: attempts})
	requireSSELimitError(t, err, sseLimitDecodeAttempts)
	if len(rec.calls) != attempts {
		t.Fatalf("callback attempts = %d, want %d", len(rec.calls), attempts)
	}
	if reader.reads > attempts+2 {
		t.Fatalf("parser kept reading after the attempt budget: reads=%d", reader.reads)
	}
}

func TestIssue152FailedPendingAtEOFIsNotDecodedAgain(t *testing.T) {
	rec := &recordingSSECallback{fn: decodeFailure}
	err := consumeSSEWithLimits(strings.NewReader("data: {\n\n"), rec.onData, sseLimits{maxEventBytes: 64, maxDecodeAttempts: 4})
	if err == nil || !isLikelyOpenAIDecodeError(err) {
		t.Fatalf("err = %v, want the retained decode error", err)
	}
	if len(rec.calls) != 1 {
		t.Fatalf("callback calls = %d, want one (no EOF re-decode of the same candidate)", len(rec.calls))
	}

	// EOF with new, unflushed data still gets one attempt within budget.
	rec = &recordingSSECallback{fn: func(data string) error {
		if data == "{\n1}" {
			return nil
		}
		return decodeFailure(data)
	}}
	if err := consumeSSEWithLimits(strings.NewReader("data: {\n\ndata: 1}"), rec.onData, sseLimits{maxEventBytes: 64, maxDecodeAttempts: 4}); err != nil {
		t.Fatalf("EOF with new data: %v", err)
	}
	if len(rec.calls) != 2 {
		t.Fatalf("calls = %q", rec.calls)
	}
}

func TestIssue152IndependentEventsDoNotShareBudget(t *testing.T) {
	const budget = 16
	var input strings.Builder
	for i := 0; i < 50; i++ {
		fmt.Fprintf(&input, "data: event-%06d\n\n", i) // 12 bytes each
	}
	input.WriteString("data: [DONE]\n\n")
	rec := &recordingSSECallback{fn: func(data string) error {
		if data == "[DONE]" {
			return errSSEDone
		}
		return nil
	}}
	err := consumeSSEWithLimits(strings.NewReader(input.String()), rec.onData, sseLimits{maxEventBytes: budget, maxDecodeAttempts: 2})
	if !errors.Is(err, errSSEDone) {
		t.Fatalf("err = %v, want the terminal marker", err)
	}
	if len(rec.calls) != 51 {
		t.Fatalf("calls = %d, want 51", len(rec.calls))
	}
}

func TestIssue152NonDecodeErrorsAndDoneReturnImmediately(t *testing.T) {
	sentinel := errors.New("fixture callback failure")
	rec := &recordingSSECallback{fn: func(string) error { return sentinel }}
	err := consumeSSEWithLimits(strings.NewReader("data: a\n\ndata: b\n\n"), rec.onData, sseLimits{maxEventBytes: 64, maxDecodeAttempts: 2})
	if !errors.Is(err, sentinel) || len(rec.calls) != 1 {
		t.Fatalf("err=%v calls=%d, want first sentinel", err, len(rec.calls))
	}

	rec = &recordingSSECallback{fn: func(string) error { return errSSEDone }}
	err = consumeSSEWithLimits(strings.NewReader("data: [DONE]\n\ndata: late\n\n"), rec.onData, sseLimits{maxEventBytes: 64, maxDecodeAttempts: 2})
	if !errors.Is(err, errSSEDone) || len(rec.calls) != 1 {
		t.Fatalf("err=%v calls=%d, want done after one call", err, len(rec.calls))
	}

	readErr := errors.New("fixture read failure")
	err = consumeSSEWithLimits(io.MultiReader(strings.NewReader("data: a\n"), iotestErrReader{readErr}), (&recordingSSECallback{}).onData, sseLimits{maxEventBytes: 64, maxDecodeAttempts: 2})
	if !errors.Is(err, readErr) {
		t.Fatalf("err = %v, want reader failure", err)
	}
}

type iotestErrReader struct{ err error }

func (r iotestErrReader) Read([]byte) (int, error) { return 0, r.err }

func TestIssue152InvalidLimitsAreRejected(t *testing.T) {
	for _, limits := range []sseLimits{{0, 1}, {1, 0}, {-1, 1}} {
		if err := consumeSSEWithLimits(strings.NewReader("data: a\n\n"), (&recordingSSECallback{}).onData, limits); err == nil {
			t.Fatalf("limits %+v accepted", limits)
		}
	}
}

func TestIssue152LimitErrorCarriesNoContentOrRetryVocabulary(t *testing.T) {
	const canary = "CANARY_SECRET_152"
	input := "data: " + canary + strings.Repeat("x", 64) + "\n\n"
	err := consumeSSEWithLimits(strings.NewReader(input), (&recordingSSECallback{}).onData, sseLimits{maxEventBytes: 16, maxDecodeAttempts: 2})
	requireSSELimitError(t, err, sseLimitEventBytes)
	msg := strings.ToLower(err.Error())
	if strings.Contains(err.Error(), canary) {
		t.Fatalf("limit error leaked stream content: %q", err.Error())
	}
	for _, word := range []string{"rate limit", "too many requests", "timeout", "timed out", "connection", "eof", "context", "token", "status", "decode", "server error", "retry", "tls", "unavailable", "unauthorized", "broken pipe", "reset by peer", "http", "overloaded"} {
		if strings.Contains(msg, word) {
			t.Fatalf("limit error %q contains classifier vocabulary %q", err.Error(), word)
		}
	}
}

// The production entry must use the default attempt budget; a malformed
// fragment stream stops after exactly the default number of callback attempts.
func TestIssue152DefaultEntryEnforcesDecodeAttemptBudget(t *testing.T) {
	input := strings.Repeat("data: {\n\n", defaultSSEMaxDecodeAttempts+8)
	rec := &recordingSSECallback{fn: decodeFailure}
	err := consumeSSE(strings.NewReader(input), rec.onData)
	requireSSELimitError(t, err, sseLimitDecodeAttempts)
	if len(rec.calls) != defaultSSEMaxDecodeAttempts {
		t.Fatalf("callback attempts = %d, want %d", len(rec.calls), defaultSSEMaxDecodeAttempts)
	}
}

// lazyLinesReader serves count copies of line without materializing them.
type lazyLinesReader struct {
	line      []byte
	remaining int
	offset    int
}

func (r *lazyLinesReader) Read(p []byte) (int, error) {
	if r.remaining == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.line[r.offset:])
	r.offset += n
	if r.offset == len(r.line) {
		r.offset = 0
		r.remaining--
	}
	return n, nil
}

func TestIssue152DefaultEntryEnforcesAggregateByteBudget(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates the default 32 MiB aggregate budget once")
	}
	const lineBytes = 1 << 20
	line := append(append([]byte("data: "), bytes.Repeat([]byte("x"), lineBytes)...), '\n')
	// 33 one-MiB data lines without a blank line: each fits the scanner's
	// single-line buffer, but the aggregate exceeds the default event budget.
	reader := &lazyLinesReader{line: line, remaining: defaultSSEMaxEventBytes/lineBytes + 1}
	rec := &recordingSSECallback{}
	err := consumeSSE(reader, rec.onData)
	requireSSELimitError(t, err, sseLimitEventBytes)
	if len(rec.calls) != 0 {
		t.Fatalf("callback ran for an over-budget aggregate")
	}
}

type countingCloseBody struct {
	io.Reader
	closes atomic.Int32
}

func (b *countingCloseBody) Close() error {
	b.closes.Add(1)
	return nil
}

func sseHTTPClient(body *countingCloseBody, requests *atomic.Int32) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		requests.Add(1)
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       body,
			Request:    r,
		}, nil
	})}
}

func assertSSELimitTerminal(t *testing.T, events []llm.StreamEvent, wantText string) {
	t.Helper()
	text := ""
	sawLimit := false
	for _, event := range events {
		switch typed := event.(type) {
		case llm.StreamDoneEvent:
			t.Fatalf("resource-limited stream emitted StreamDoneEvent: %#v", events)
		case llm.StreamTextDeltaEvent:
			text += typed.Delta
		case llm.StreamErrorEvent:
			var limitErr *sseResourceLimitError
			if !errors.As(typed.AsError(), &limitErr) {
				t.Fatalf("stream error = %v, want SSE resource limit", typed.AsError())
			}
			sawLimit = true
		}
	}
	if !sawLimit {
		t.Fatalf("no resource-limit terminal in %#v", events)
	}
	if text != wantText {
		t.Fatalf("consumed prefix = %q, want %q", text, wantText)
	}
}

func TestIssue152ChatStreamLimit(t *testing.T) {
	body := &countingCloseBody{Reader: strings.NewReader(
		`data: {"id":"resp_1","choices":[{"delta":{"content":"partial"}}]}` + "\n\n" +
			strings.Repeat("data: {\"choices\":\n\n", defaultSSEMaxDecodeAttempts+4) +
			"data: [DONE]\n\n")}
	var requests atomic.Int32
	client := &ChatClient{HTTPClient: sseHTTPClient(body, &requests), BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	assertSSELimitTerminal(t, collectOpenAIStream(stream), "partial")
	if got := body.closes.Load(); got != 1 {
		t.Fatalf("body closes = %d, want 1", got)
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("HTTP requests = %d, want 1", got)
	}
}

func TestIssue152ResponsesStreamLimit(t *testing.T) {
	body := &countingCloseBody{Reader: strings.NewReader(
		`data: {"type":"response.output_text.delta","delta":"partial"}` + "\n\n" +
			strings.Repeat("data: {\"type\":\n\n", defaultSSEMaxDecodeAttempts+4) +
			`data: {"type":"response.completed","response":{"status":"completed"}}` + "\n\n")}
	var requests atomic.Int32
	client := &ResponsesClient{HTTPClient: sseHTTPClient(body, &requests), BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	assertSSELimitTerminal(t, collectOpenAIStream(stream), "partial")
	if got := body.closes.Load(); got != 1 {
		t.Fatalf("body closes = %d, want 1", got)
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("HTTP requests = %d, want 1", got)
	}
}

func TestIssue152ChatStreamLegitimateSplitJSONStillDecodes(t *testing.T) {
	// A gateway that inserts a premature blank line inside one JSON event keeps
	// working within the attempt budget.
	body := &countingCloseBody{Reader: strings.NewReader(
		"data: {\"choices\":[{\"delta\":\n\n" +
			"data: {\"content\":\"joined\"}}]}\n\n" +
			"data: [DONE]\n\n")}
	var requests atomic.Int32
	client := &ChatClient{HTTPClient: sseHTTPClient(body, &requests), BaseURL: "https://example.com", ModelName: "test-model", MaxRetries: 1}
	stream, err := client.InvokeStream(context.Background(), llm.InvokeRequest{Messages: []llm.Message{llm.NewUserMessage("hello")}})
	if err != nil {
		t.Fatal(err)
	}
	text := ""
	done := false
	for _, event := range collectOpenAIStream(stream) {
		switch typed := event.(type) {
		case llm.StreamTextDeltaEvent:
			text += typed.Delta
		case llm.StreamDoneEvent:
			done = true
		case llm.StreamErrorEvent:
			t.Fatalf("unexpected error: %v", typed.AsError())
		}
	}
	if !done || text != "joined" {
		t.Fatalf("done=%v text=%q", done, text)
	}
}
