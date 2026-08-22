package compaction

import (
	"context"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
)

// ArtifactOwnerProvider resolves the execution subject that owns compaction
// artifacts. The reducer adds the role/tool-call qualifiers for each object.
type ArtifactOwnerProvider func(context.Context) (artifact.Owner, error)

type ArtifactRequest struct {
	SessionID  string
	MessageKey string
	PartKey    string
	ToolName   string
	ToolCallID string
	Content    string
}

type ArtifactResult struct {
	Path string
}

type ArtifactWriter interface {
	SaveCompactionArtifact(ctx context.Context, req ArtifactRequest) (ArtifactResult, error)
}

type ArtifactWriterFunc func(context.Context, ArtifactRequest) (ArtifactResult, error)

func (f ArtifactWriterFunc) SaveCompactionArtifact(ctx context.Context, req ArtifactRequest) (ArtifactResult, error) {
	if f == nil {
		return ArtifactResult{}, nil
	}
	return f(ctx, req)
}
