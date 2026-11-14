package proxy

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/codex2api/auth"
	"github.com/codex2api/config"
	"github.com/codex2api/database"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

const (
	geminiStreamTestThought = `{"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"THOUGHT_ONLY","thought":true}]}}]}}`
	geminiStreamTestAnswer  = `{"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ANSWER TEST_END"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"thoughtsTokenCount":5,"totalTokenCount":35}}}`
	geminiStreamTestRequest = `{"contents":[{"role":"user","parts":[{"text":"give ANSWER TEST_END"}]}]}`
)

func newGeminiStreamTestHandler(t *testing.T, upstreamHandler http.HandlerFunc) (*gin.Engine, *database.DB) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	db, err := database.New("sqlite", filepath.Join(t.TempDir(), "gemini-stream.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	db.SetUsageLogConfig(database.UsageLogModeFull, 100, 60)
	keyID, err := db.InsertAPIKeyWithOptions(context.Background(), database.APIKeyInput{
		Name: "Gemini stream test", Key: "gemini-stream-test-key",
		Limits: database.APIKeyLimits{UpstreamChannel: database.UpstreamChannelAntigravity},
	})
	if err != nil {
		t.Fatal(err)
	}
	key, err := db.GetAPIKeyByID(context.Background(), keyID)
	if err != nil {
		t.Fatal(err)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		upstreamHandler(w, r)
	}))
	t.Cleanup(func() {
		upstream.CloseClientConnections()
		upstream.Close()
	})
	previous := antigravityOAuthEndpointBases
	antigravityOAuthEndpointBases = []string{upstream.URL}
	t.Cleanup(func() { antigravityOAuthEndpointBases = previous })

	store := auth.NewStore(nil, nil, &database.SystemSettings{MaxConcurrency: 2, MaxRetries: 0})
	t.Cleanup(store.Stop)
	store.AddAccount(&auth.Account{
		DBID: 72301, UpstreamType: auth.UpstreamAntigravity,
		AccessToken: "test-token", AntigravityProjectID: "test-project",
		Models: []string{"gemini-3.7-flash-tiered"}, HealthTier: auth.HealthTierHealthy, Status: auth.StatusReady,
	})
	handler := NewHandler(store, db, &config.Config{AllowAnonymousV1: true}, nil)
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set(contextAPIKeyRow, key)
		c.Next()
	})
	router.POST("/v1beta/models/*action", handler.GeminiModelsAction)
	return router, db
}

func geminiStreamTestHTTPRequest(ctx context.Context, method string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-3.7-flash-low:"+method+"?alt=sse", strings.NewReader(geminiStreamTestRequest)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	return req
}

func assertGeminiStreamUsageStatus(t *testing.T, db *database.DB, want int, stream bool) database.UsageLog {
	t.Helper()
	db.FlushUsageLogs()
	logs, err := db.ListUsageLogsByFilter(context.Background(), database.UsageLogFilter{
		Start: time.Now().Add(-time.Minute), End: time.Now().Add(time.Minute), IncludeCanceled: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(logs) != 1 {
		t.Fatalf("usage logs = %d, want 1: %+v", len(logs), logs)
	}
	if entry := logs[0]; entry.StatusCode != want || entry.Stream != stream {
		t.Fatalf("usage status=%d stream=%v, want status=%d stream=%v", entry.StatusCode, entry.Stream, want, stream)
	}
	return *logs[0]
}

func TestGeminiNativeStreamFlushesThoughtAndKeepsAliveBeforeAnswer(t *testing.T) {
	previousInterval := continuousRetryKeepaliveInterval
	continuousRetryKeepaliveInterval = 30 * time.Millisecond
	t.Cleanup(func() { continuousRetryKeepaliveInterval = previousInterval })
	allowAnswer := make(chan struct{})
	var answerOnce sync.Once
	releaseAnswer := func() { answerOnce.Do(func() { close(allowAnswer) }) }
	t.Cleanup(releaseAnswer)
	upstreamCanceled := make(chan struct{})
	router, db := newGeminiStreamTestHandler(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+geminiStreamTestThought+"\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-allowAnswer:
			_, _ = io.WriteString(w, "data: "+geminiStreamTestAnswer+"\n\n")
			w.(http.Flusher).Flush()
		case <-r.Context().Done():
			close(upstreamCanceled)
		}
	})
	handlerDone := make(chan struct{})
	downstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		router.ServeHTTP(w, r)
		close(handlerDone)
	}))
	t.Cleanup(downstream.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, downstream.URL+"/v1beta/models/gemini-3.7-flash-low:streamGenerateContent?alt=sse", strings.NewReader(geminiStreamTestRequest))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := downstream.Client().Do(req)
	if err != nil {
		t.Fatalf("thought frame was not flushed while the upstream answer was blocked: %v", err)
	}
	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)
	for {
		first, err := reader.ReadString('\n')
		if err != nil {
			t.Fatalf("read thought before releasing answer: %v", err)
		}
		if !strings.HasPrefix(first, "data: ") {
			continue
		}
		if strings.TrimSpace(first) == "data: {}" {
			continue
		}
		if !strings.Contains(first, "THOUGHT_ONLY") || !strings.Contains(first, `"thought":true`) {
			t.Fatalf("first frame=%q, want thought before releasing answer", first)
		}
		break
	}
	// The upstream remains silent until the client receives a heartbeat. This
	// proves keepalive continues during body reads, after the headers and thought.
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatalf("waiting for keepalive while the upstream answer is blocked: %v", err)
		}
		if strings.TrimSpace(line) == "data: {}" {
			break
		}
		if strings.HasPrefix(line, "data: ") {
			t.Fatalf("unexpected generation frame before releasing answer: %s", line)
		}
	}
	select {
	case <-upstreamCanceled:
		t.Fatal("upstream was canceled before the answer")
	default:
	}
	releaseAnswer()
	rest, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(rest), "ANSWER TEST_END") || !strings.Contains(string(rest), `"finishReason":"STOP"`) {
		t.Fatalf("status=%d rest=%s, want complete native answer", resp.StatusCode, rest)
	}
	select {
	case <-handlerDone:
	case <-time.After(time.Second):
		t.Fatal("handler did not release the completed stream")
	}
	assertGeminiStreamUsageStatus(t, db, http.StatusOK, true)
}

func TestGeminiNativeStreamFailureIsNotSuccessfulStop(t *testing.T) {
	for _, tc := range []struct {
		name      string
		truncated bool
	}{
		{name: "EOF without terminal"},
		{name: "HTTP body interrupted", truncated: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, db := newGeminiStreamTestHandler(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				if tc.truncated {
					w.Header().Set("Content-Length", "10000")
				}
				_, _ = io.WriteString(w, "data: "+geminiStreamTestThought+"\n\n")
				w.(http.Flusher).Flush()
			})
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, geminiStreamTestHTTPRequest(context.Background(), "streamGenerateContent"))
			body := recorder.Body.String()
			if recorder.Code != http.StatusOK || !strings.Contains(body, "THOUGHT_ONLY") {
				t.Fatalf("status=%d body=%s, want partial content followed by a stream error", recorder.Code, body)
			}
			if strings.Contains(body, `"finishReason":"STOP"`) {
				t.Fatalf("failed stream was reported as successful STOP: %s", body)
			}
			// The Google SDK expects a bare JSON error after the SSE data
			// events. A data-prefixed error is silently discarded by older SDKs.
			errorOffset := strings.Index(body, "\n\n{\"error\":")
			if errorOffset < 0 {
				t.Fatalf("missing native Gemini stream error: %s", body)
			}
			payload := body[errorOffset+2:]
			if !gjson.Valid(payload) || gjson.Get(payload, "error.code").Int() != http.StatusBadGateway || gjson.Get(payload, "error.status").String() != "UNAVAILABLE" {
				t.Fatalf("invalid native Gemini stream error: %s", body)
			}
			assertGeminiStreamUsageStatus(t, db, logStatusUpstreamStreamBreak, true)
		})
	}
}

func TestGeminiNativeStreamFailureBeforeContentReturnsHTTPError(t *testing.T) {
	router, db := newGeminiStreamTestHandler(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
	})
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, geminiStreamTestHTTPRequest(context.Background(), "streamGenerateContent"))
	body := recorder.Body.String()
	if recorder.Code != http.StatusBadGateway || gjson.Get(body, "error.code").Int() != http.StatusBadGateway || gjson.Get(body, "error.status").String() != "UNAVAILABLE" {
		t.Fatalf("status=%d body=%s, want Gemini HTTP 502 error", recorder.Code, body)
	}
	assertGeminiStreamUsageStatus(t, db, logStatusUpstreamStreamBreak, true)
}

type geminiStreamDisconnectWriter struct {
	*httptest.ResponseRecorder
	cancel context.CancelFunc
	fail   bool
	wrote  bool
}

func (w *geminiStreamDisconnectWriter) Write(p []byte) (int, error) {
	w.wrote = true
	if w.fail {
		return 0, errors.New("test downstream disconnected")
	}
	n, err := w.ResponseRecorder.Write(p)
	w.cancel()
	return n, err
}

func (w *geminiStreamDisconnectWriter) WriteString(s string) (int, error) {
	return w.Write([]byte(s))
}

func TestGeminiNativeStreamDownstreamDisconnectReleasesUpstream(t *testing.T) {
	for _, failWrite := range []bool{false, true} {
		name := "client cancellation"
		if failWrite {
			name = "downstream write failure"
		}
		t.Run(name, func(t *testing.T) {
			upstreamCanceled := make(chan struct{})
			router, db := newGeminiStreamTestHandler(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: "+geminiStreamTestThought+"\n\n")
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				close(upstreamCanceled)
			})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			writer := &geminiStreamDisconnectWriter{ResponseRecorder: httptest.NewRecorder(), cancel: cancel, fail: failWrite}
			handlerDone := make(chan struct{})
			go func() {
				router.ServeHTTP(writer, geminiStreamTestHTTPRequest(ctx, "streamGenerateContent"))
				close(handlerDone)
			}()
			select {
			case <-handlerDone:
			case <-time.After(2 * time.Second):
				t.Fatal("handler did not terminate after downstream disconnect")
			}
			if !writer.wrote {
				t.Fatal("fixture did not reach downstream write")
			}
			select {
			case <-upstreamCanceled:
			case <-time.After(time.Second):
				t.Fatal("upstream request was not canceled after downstream disconnect")
			}
			assertGeminiStreamUsageStatus(t, db, logStatusClientClosed, true)
		})
	}
}

func TestGeminiNativeNonStreamStillReturnsCompleteAnswer(t *testing.T) {
	router, db := newGeminiStreamTestHandler(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, geminiStreamTestAnswer)
	})
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, geminiStreamTestHTTPRequest(context.Background(), "generateContent"))
	body := recorder.Body.String()
	if recorder.Code != http.StatusOK || gjson.Get(body, "candidates.0.content.parts.0.text").String() != "ANSWER TEST_END" || gjson.Get(body, "candidates.0.finishReason").String() != "STOP" {
		t.Fatalf("status=%d body=%s, want complete native answer", recorder.Code, body)
	}
	if gjson.Get(body, "response").Exists() {
		t.Fatalf("upstream envelope was not removed: %s", body)
	}
	assertGeminiStreamUsageStatus(t, db, http.StatusOK, false)
}

func newGeminiStreamMultiAccountTestHandler(t *testing.T, maxRetries int, upstreamHandler http.HandlerFunc) (*gin.Engine, *auth.Store, *database.DB) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	db, err := database.New("sqlite", filepath.Join(t.TempDir(), "gemini-ttft.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	db.SetUsageLogConfig(database.UsageLogModeFull, 100, 60)
	keyID, err := db.InsertAPIKeyWithOptions(context.Background(), database.APIKeyInput{
		Name: "Gemini TTFT test", Key: "gemini-ttft-test-key",
		Limits: database.APIKeyLimits{UpstreamChannel: database.UpstreamChannelAntigravity},
	})
	if err != nil {
		t.Fatal(err)
	}
	key, err := db.GetAPIKeyByID(context.Background(), keyID)
	if err != nil {
		t.Fatal(err)
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		upstreamHandler(w, r)
	}))
	t.Cleanup(func() {
		upstream.CloseClientConnections()
		upstream.Close()
	})
	previousBases := antigravityOAuthEndpointBases
	antigravityOAuthEndpointBases = []string{upstream.URL}
	t.Cleanup(func() { antigravityOAuthEndpointBases = previousBases })

	store := auth.NewStore(nil, nil, &database.SystemSettings{MaxConcurrency: 2, MaxRetries: maxRetries})
	store.SetMaxRetries(maxRetries)
	t.Cleanup(store.Stop)
	store.AddAccount(&auth.Account{
		DBID: 8414, UpstreamType: auth.UpstreamAntigravity,
		AccessToken: "token-8414", AntigravityProjectID: "proj-8414",
		Models: []string{"gemini-3.7-flash-tiered"}, HealthTier: auth.HealthTierHealthy, Status: auth.StatusReady,
	})
	store.AddAccount(&auth.Account{
		DBID: 8415, UpstreamType: auth.UpstreamAntigravity,
		AccessToken: "token-8415", AntigravityProjectID: "proj-8415",
		Models: []string{"gemini-3.7-flash-tiered"}, HealthTier: auth.HealthTierHealthy, Status: auth.StatusReady,
	})
	handler := NewHandler(store, db, &config.Config{AllowAnonymousV1: true}, nil)
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set(contextAPIKeyRow, key)
		c.Next()
	})
	router.POST("/v1beta/models/*action", handler.GeminiModelsAction)
	return router, store, db
}

func TestGeminiNativeStreamFirstTokenTimeoutRotatesAccount(t *testing.T) {
	previousTimeout := geminiFirstTokenTimeoutFor
	geminiFirstTokenTimeoutFor = func() time.Duration { return 35 * time.Millisecond }
	t.Cleanup(func() { geminiFirstTokenTimeoutFor = previousTimeout })

	for _, tc := range []struct {
		name         string
		sendHeaders  bool
		keepaliveInt time.Duration
	}{
		{name: "hangs before response headers", sendHeaders: false},
		{name: "hangs after 200 SSE headers", sendHeaders: true},
		{name: "hangs after 200 SSE headers with keepalive committed", sendHeaders: true, keepaliveInt: 10 * time.Millisecond},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.keepaliveInt > 0 {
				previousInterval := continuousRetryKeepaliveInterval
				continuousRetryKeepaliveInterval = tc.keepaliveInt
				t.Cleanup(func() { continuousRetryKeepaliveInterval = previousInterval })
			}
			var mu sync.Mutex
			seenTokens := make([]string, 0, 2)
			firstCanceled := make(chan struct{})
			var cancelOnce sync.Once
			router, store, db := newGeminiStreamMultiAccountTestHandler(t, 1, func(w http.ResponseWriter, r *http.Request) {
				authHeader := r.Header.Get("Authorization")
				mu.Lock()
				attempt := len(seenTokens)
				seenTokens = append(seenTokens, authHeader)
				mu.Unlock()
				if attempt == 0 {
					if tc.sendHeaders {
						w.Header().Set("Content-Type", "text/event-stream")
						w.WriteHeader(http.StatusOK)
						w.(http.Flusher).Flush()
					}
					<-r.Context().Done()
					cancelOnce.Do(func() { close(firstCanceled) })
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: "+geminiStreamTestAnswer+"\n\n")
				w.(http.Flusher).Flush()
			})

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, geminiStreamTestHTTPRequest(ctx, "streamGenerateContent"))
			if recorder.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s, want 200 after rotating account", recorder.Code, recorder.Body.String())
			}
			body := recorder.Body.String()
			if !strings.Contains(body, "ANSWER TEST_END") || !strings.Contains(body, `"finishReason":"STOP"`) {
				t.Fatalf("missing retried answer in body: %s", body)
			}
			select {
			case <-firstCanceled:
			case <-time.After(time.Second):
				t.Fatal("first stalled upstream attempt was not canceled by TTFT guard")
			}
			mu.Lock()
			if len(seenTokens) != 2 || seenTokens[0] == seenTokens[1] {
				mu.Unlock()
				t.Fatalf("expected rotation across 2 distinct accounts, got tokens=%v", seenTokens)
			}
			firstToken := seenTokens[0]
			mu.Unlock()

			// Soft TTFT exclusion should not penalize the timed-out account's health.
			firstAccountID := int64(8414)
			if firstToken == "Bearer token-8415" {
				firstAccountID = 8415
			}
			if acc := store.FindByID(firstAccountID); acc == nil || acc.HealthTier != auth.HealthTierHealthy {
				t.Fatalf("timed-out account %d penalized: %+v", firstAccountID, acc)
			}
			entry := assertGeminiStreamUsageStatus(t, db, http.StatusOK, true)
			if entry.AttemptIndex != 2 {
				t.Fatalf("usage AttemptIndex = %d, want 2", entry.AttemptIndex)
			}
		})
	}
}

func TestGeminiNativeStreamThoughtFrameDisarmsFirstTokenTimeout(t *testing.T) {
	previousTimeout := geminiFirstTokenTimeoutFor
	geminiFirstTokenTimeoutFor = func() time.Duration { return 35 * time.Millisecond }
	t.Cleanup(func() { geminiFirstTokenTimeoutFor = previousTimeout })

	router, db := newGeminiStreamTestHandler(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+geminiStreamTestThought+"\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-time.After(90 * time.Millisecond):
			_, _ = io.WriteString(w, "data: "+geminiStreamTestAnswer+"\n\n")
			w.(http.Flusher).Flush()
		case <-r.Context().Done():
		}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, geminiStreamTestHTTPRequest(ctx, "streamGenerateContent"))
	body := recorder.Body.String()
	if recorder.Code != http.StatusOK || !strings.Contains(body, "THOUGHT_ONLY") || !strings.Contains(body, "ANSWER TEST_END") {
		t.Fatalf("thought frame did not disarm TTFT guard: status=%d body=%s", recorder.Code, body)
	}
	assertGeminiStreamUsageStatus(t, db, http.StatusOK, true)
}

func TestGeminiNativeStreamFirstTokenTimeoutExhaustedReturns504(t *testing.T) {
	previousTimeout := geminiFirstTokenTimeoutFor
	geminiFirstTokenTimeoutFor = func() time.Duration { return 30 * time.Millisecond }
	t.Cleanup(func() { geminiFirstTokenTimeoutFor = previousTimeout })

	for _, sendHeaders := range []bool{false, true} {
		name := "before headers"
		if sendHeaders {
			name = "after 200 SSE headers"
		}
		t.Run(name, func(t *testing.T) {
			router, store, _ := newGeminiStreamMultiAccountTestHandler(t, 0, func(w http.ResponseWriter, r *http.Request) {
				if sendHeaders {
					w.Header().Set("Content-Type", "text/event-stream")
					w.WriteHeader(http.StatusOK)
					w.(http.Flusher).Flush()
				}
				<-r.Context().Done()
			})
			_ = store
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, geminiStreamTestHTTPRequest(ctx, "streamGenerateContent"))
			body := recorder.Body.String()
			if recorder.Code != http.StatusGatewayTimeout ||
				gjson.Get(body, "error.code").Int() != http.StatusGatewayTimeout ||
				gjson.Get(body, "error.status").String() != "DEADLINE_EXCEEDED" {
				t.Fatalf("status=%d body=%s, want Gemini 504 DEADLINE_EXCEEDED", recorder.Code, body)
			}
		})
	}
}
