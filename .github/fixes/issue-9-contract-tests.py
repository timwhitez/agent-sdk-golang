from pathlib import Path

chat_test = Path("sdk/llm/openai/chat_test.go")
text = chat_test.read_text(encoding="utf-8")
for old, new in [
    ("\t\t{code: 401, want: true},", "\t\t{code: 401, want: false},"),
    ("\t\t{code: 403, want: true},", "\t\t{code: 403, want: false},"),
]:
    if text.count(old) != 1:
        raise SystemExit(f"unexpected OpenAI retry assertion count for {old!r}: {text.count(old)}")
    text = text.replace(old, new, 1)
chat_test.write_text(text, encoding="utf-8")

agent_test = Path("sdk/agent/agent_retry_loop_guard_test.go")
text = agent_test.read_text(encoding="utf-8")
for entry in [
    '\t\t{name: "auth_401", err: errors.New("provider failed with HTTP status 401 unauthorized")},\n',
    '\t\t{name: "permission_403", err: errors.New("provider failed with HTTP status 403 permission denied")},\n',
]:
    if text.count(entry) != 1:
        raise SystemExit(f"unexpected transient auth fixture count for {entry!r}: {text.count(entry)}")
    text = text.replace(entry, "", 1)
anchor = '\t\t{name: "bad_request_400", err: errors.New("provider failed with HTTP status 400 invalid request")},\n'
addition = (
    '\t\t{name: "auth_401", err: errors.New("provider failed with HTTP status 401 unauthorized")},\n'
    '\t\t{name: "permission_403", err: errors.New("provider failed with HTTP status 403 permission denied")},\n'
    + anchor
)
if text.count(anchor) != 1:
    raise SystemExit(f"unexpected non-retryable fixture anchor count: {text.count(anchor)}")
text = text.replace(anchor, addition, 1)
agent_test.write_text(text, encoding="utf-8")
