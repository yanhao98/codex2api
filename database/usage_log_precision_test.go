package database

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestUsageLogPrecisionCreatedAt_SQLite(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "precision_test.db")
	db, err := New("sqlite", dbPath)
	if err != nil {
		t.Fatalf("New(sqlite) error: %v", err)
	}
	// Stop background flusher to manually control flush timing
	close(db.logStop)
	db.logWg.Wait()
	defer db.conn.Close()

	ctx := context.Background()

	// 1. Explicit CreatedAt test
	explicitTime := time.Date(2026, 11, 4, 15, 30, 45, 0, time.UTC)
	if err := db.InsertUsageLog(ctx, &UsageLogInput{
		Endpoint:    "/v1/chat/completions",
		Model:       "gemini-3.8-flash",
		StatusCode:  200,
		TotalTokens: 100,
		CreatedAt:   explicitTime,
	}); err != nil {
		t.Fatalf("InsertUsageLog with explicit CreatedAt error: %v", err)
	}

	// 2. Implicit zero CreatedAt test (should snapshot time at call time)
	tBefore := time.Now().UTC().Truncate(time.Second)
	time.Sleep(10 * time.Millisecond)
	if err := db.InsertUsageLog(ctx, &UsageLogInput{
		Endpoint:    "/v1/chat/completions",
		Model:       "claude-opus-5-5-high",
		StatusCode:  200,
		TotalTokens: 200,
	}); err != nil {
		t.Fatalf("InsertUsageLog with zero CreatedAt error: %v", err)
	}
	time.Sleep(10 * time.Millisecond)
	tAfter := time.Now().UTC().Add(time.Second)

	// Simulate batch queue delay before flush
	time.Sleep(200 * time.Millisecond)
	flushTime := time.Now().UTC()

	// Flush buffered logs
	db.FlushUsageLogs()

	// Query inserted logs
	rows, err := db.conn.QueryContext(ctx, "SELECT model, created_at FROM usage_logs ORDER BY id ASC")
	if err != nil {
		t.Fatalf("Query usage_logs error: %v", err)
	}
	defer rows.Close()

	type result struct {
		model     string
		createdAt string
	}
	var results []result
	for rows.Next() {
		var r result
		if err := rows.Scan(&r.model, &r.createdAt); err != nil {
			t.Fatalf("Scan error: %v", err)
		}
		results = append(results, r)
	}

	if len(results) != 2 {
		t.Fatalf("got %d rows, want 2", len(results))
	}

	parseTime := func(str string) (time.Time, error) {
		for _, layout := range []string{time.RFC3339, time.RFC3339Nano, "2006-01-02 15:04:05"} {
			if t, err := time.Parse(layout, str); err == nil {
				return t.UTC(), nil
			}
		}
		return time.Time{}, fmt.Errorf("unable to parse %q", str)
	}

	row1Time, err := parseTime(results[0].createdAt)
	if err != nil {
		t.Fatalf("Parse row 1 created_at error: %v", err)
	}
	if !row1Time.Equal(explicitTime) {
		t.Errorf("Row 1 created_at %v != want %v", row1Time, explicitTime)
	}

	row2Time, err := parseTime(results[1].createdAt)
	if err != nil {
		t.Fatalf("Parse row 2 created_at error: %v", err)
	}
	if row2Time.Before(tBefore) || row2Time.After(tAfter) {
		t.Errorf("Row 2 created_at %v not between %v and %v (flushTime was %v)", row2Time, tBefore, tAfter, flushTime)
	}
}

func TestUsageLogPrecisionCreatedAt_Postgres(t *testing.T) {
	dsn := os.Getenv("CODEX2API_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("CODEX2API_TEST_POSTGRES_DSN not set, skipping postgres precision test")
	}

	db, err := New("postgres", dsn)
	if err != nil {
		t.Fatalf("New(postgres) error: %v", err)
	}
	close(db.logStop)
	db.logWg.Wait()
	defer db.conn.Close()

	ctx := context.Background()

	// Clean table for testing
	if _, err := db.conn.ExecContext(ctx, "DELETE FROM usage_logs"); err != nil {
		t.Fatalf("Delete error: %v", err)
	}

	explicitTime := time.Date(2026, 11, 4, 15, 30, 45, 123456000, time.UTC)
	if err := db.InsertUsageLog(ctx, &UsageLogInput{
		Endpoint:    "/v1/chat/completions",
		Model:       "gemini-3.8-flash",
		StatusCode:  200,
		TotalTokens: 100,
		CreatedAt:   explicitTime,
	}); err != nil {
		t.Fatalf("InsertUsageLog with explicit CreatedAt error: %v", err)
	}

	tBefore := time.Now().UTC()
	time.Sleep(5 * time.Millisecond)
	if err := db.InsertUsageLog(ctx, &UsageLogInput{
		Endpoint:    "/v1/chat/completions",
		Model:       "claude-opus-5-5-high",
		StatusCode:  200,
		TotalTokens: 200,
	}); err != nil {
		t.Fatalf("InsertUsageLog with zero CreatedAt error: %v", err)
	}
	time.Sleep(5 * time.Millisecond)
	tAfter := time.Now().UTC()

	// Wait before flushing to simulate batching
	time.Sleep(500 * time.Millisecond)

	db.FlushUsageLogs()

	rows, err := db.conn.QueryContext(ctx, "SELECT model, created_at FROM usage_logs ORDER BY id ASC")
	if err != nil {
		t.Fatalf("Query error: %v", err)
	}
	defer rows.Close()

	var gotModels []string
	var gotTimes []time.Time
	for rows.Next() {
		var m string
		var ts time.Time
		if err := rows.Scan(&m, &ts); err != nil {
			t.Fatalf("Scan error: %v", err)
		}
		gotModels = append(gotModels, m)
		gotTimes = append(gotTimes, ts.UTC())
	}

	if len(gotModels) != 2 {
		t.Fatalf("got %d rows, want 2", len(gotModels))
	}

	// Verify row 1: exact microsecond precision matching explicitTime
	if !gotTimes[0].Equal(explicitTime) {
		t.Errorf("Postgres Row 1 created_at = %v, want %v", gotTimes[0], explicitTime)
	}

	// Verify row 2: call time snapshot
	if gotTimes[1].Before(tBefore.Add(-time.Second)) || gotTimes[1].After(tAfter.Add(time.Second)) {
		t.Errorf("Postgres Row 2 created_at %v not near call time [%v, %v]", gotTimes[1], tBefore, tAfter)
	}
}
