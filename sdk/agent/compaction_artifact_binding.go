package agent

import (
	"context"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
)

// bindCompactionArtifactConfig keeps local compaction on the same canonical
// host binding as the Agent tool-result boundary. It deliberately replaces any
// compaction-local binding so a stale parent/session owner cannot survive an
// Agent reconfiguration.
func bindCompactionArtifactConfig(
	cfg *compaction.Config,
	warningf func(string, ...any),
	owner artifact.Owner,
	ownerProvider ArtifactOwnerProvider,
	sink artifact.Sink,
	resolver artifact.Resolver,
	capability artifact.ResolverCapability,
	codec artifact.EnvelopeCodec,
) *compaction.Config {
	if cfg == nil {
		return nil
	}
	out := *cfg
	if cfg.ProtectedTools != nil {
		out.ProtectedTools = append([]string(nil), cfg.ProtectedTools...)
	}
	if out.Warningf == nil {
		out.Warningf = warningf
	}
	// A codec only controls envelope serialization; Agent.New installs the JSON
	// codec by default even for embedders that use only the legacy
	// ToolArtifactWriter. Do not treat that default as a partial canonical host
	// binding or legacy local reduction would fail closed on a missing sink.
	canonicalConfigured := sink != nil || resolver != nil || capability.Registered ||
		strings.TrimSpace(capability.Recovery.Capability) != ""
	if !canonicalConfigured {
		out.ArtifactOwnerProvider = nil
		out.ArtifactSink = nil
		out.ArtifactResolver = nil
		out.ArtifactResolverCapability = artifact.ResolverCapability{}
		out.ArtifactEnvelopeCodec = nil
		return &out
	}
	if ownerProvider != nil {
		out.ArtifactOwnerProvider = func(ctx context.Context) (artifact.Owner, error) {
			return ownerProvider(ctx)
		}
	} else {
		out.ArtifactOwnerProvider = func(context.Context) (artifact.Owner, error) {
			return owner, nil
		}
	}
	out.ArtifactSink = sink
	out.ArtifactResolver = resolver
	out.ArtifactResolverCapability = artifact.ResolverCapability{
		Registered: capability.Registered,
		Recovery:   cloneArtifactRecovery(capability.Recovery),
	}
	out.ArtifactEnvelopeCodec = codec
	return &out
}
