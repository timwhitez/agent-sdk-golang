package sandbox

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type todoCancelConfirmer struct{ cancel context.CancelFunc }

func (c todoCancelConfirmer) Confirm(context.Context, string, string) (bool, error) {
	if c.cancel != nil {
		c.cancel()
	}
	return true, nil
}

func TestTodoWriteChecksCancellationAfterConfirmation(t *testing.T) {
	for _, name := range []string{"todowrite", "todo_write"} {
		t.Run(name, func(t *testing.T) {
			original := []TodoItem{{Content: "original", Status: "pending"}}
			s, err := New(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			s.ReplaceTodos(original)
			deps := tools.NewContainer()
			tools.Provide(deps, Key, func(context.Context) (*Sandbox, error) { return s, nil })
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			tools.Provide(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return todoCancelConfirmer{cancel}, nil })
			out, err := todoWriteToolNamed(name).Execute(ctx, `{"todos":[{"content":"replacement","status":"completed"}]}`, deps)
			if !errors.Is(err, context.Canceled) || !reflect.DeepEqual(s.TodosSnapshot(), original) {
				t.Fatalf("err=%v output=%q state=%+v", err, out.PlainText(), s.TodosSnapshot())
			}
			tools.Override(deps, ConfirmKey, func(context.Context) (Confirmer, error) { return todoCancelConfirmer{}, nil })
			_, err = todoWriteToolNamed(name).Execute(context.Background(), `{"todos":[]}`, deps)
			if err != nil || len(s.TodosSnapshot()) != 0 {
				t.Fatalf("active write err=%v state=%+v", err, s.TodosSnapshot())
			}
		})
	}
}
