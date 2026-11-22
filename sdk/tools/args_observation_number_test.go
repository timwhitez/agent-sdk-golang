package tools

import (
	"context"
	"encoding/json"
	"testing"
)

func TestPreparedCallPreservesNumericObservation(t *testing.T) {
	type args struct {
		ID int64 `json:"id"`
	}
	var seen int64
	tool := Func[args]("lookup", "", func(_ context.Context, a args, _ *Container) (any, error) { seen = a.ID; return nil, nil })
	const raw = `{"id":9007199254740993}`
	p, observation := tool.PrepareCall(raw)
	if _, err := p.Execute(context.Background(), nil); err != nil || seen != 9007199254740993 {
		t.Fatalf("execution: id=%d, err=%v", seen, err)
	}
	display, err := json.Marshal(observation.Display)
	if err != nil {
		t.Fatal(err)
	}
	if string(display) != raw {
		t.Fatalf("display=%s; handler id=%d", display, seen)
	}
}
func TestToolAcceptsRawHugeJSONNumber(t *testing.T) {
	type args struct {
		N json.RawMessage `json:"n"`
	}
	var seen string
	tool := Func[args]("measure", "", func(_ context.Context, a args, _ *Container) (any, error) { seen = string(a.N); return nil, nil })
	const raw = `{"n":1e400}`
	var control args
	if err := json.Unmarshal([]byte(raw), &control); err != nil || string(control.N) != "1e400" {
		t.Fatal("invalid fixture")
	}
	_, err := tool.Execute(context.Background(), raw, nil)
	if err != nil || seen != "1e400" {
		t.Fatalf("handler value=%q, execution error=%v", seen, err)
	}
}
