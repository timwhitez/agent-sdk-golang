from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_test.go")
text = path.read_text(encoding="utf-8")
old = '''\torigDo := webfetchDoRequest
\twebfetchDoRequest = func(_ *http.Client, _ *http.Request) (*http.Response, error) {
\t\tt.Fatalf("webfetch request should not run for blocked private destination")
\t\treturn nil, nil
\t}
\tt.Cleanup(func() { webfetchDoRequest = origDo })
'''
new = '''\torigDial := webfetchDialContext
\twebfetchDialContext = func(context.Context, string, string) (net.Conn, error) {
\t\tt.Fatalf("low-level socket dial should not run for blocked private destination")
\t\treturn nil, nil
\t}
\tt.Cleanup(func() { webfetchDialContext = origDial })
'''
if text.count(old) != 1:
    raise SystemExit(f"unexpected private-destination request seam count: {text.count(old)}")
text = text.replace(old, new, 1)

old = '''\ttests := []struct {
\t\tname          string
\t\turl           string
\t\twantSubstring string
\t}{
\t\t{name: "loopback", url: "http://127.0.0.1:8080", wantSubstring: "loopback"},
\t\t{name: "rfc1918", url: "http://10.1.2.3", wantSubstring: "private"},
\t\t{name: "link-local", url: "http://169.254.10.20", wantSubstring: "link-local"},
\t\t{name: "private-dns", url: "http://internal.test/private", wantSubstring: "private"},
\t}
'''
new = '''\ttests := []struct {
\t\tname          string
\t\turl           string
\t\twantSubstring string
\t\twantStage     string
\t}{
\t\t{name: "loopback", url: "http://127.0.0.1:8080", wantSubstring: "loopback", wantStage: "blocked request target"},
\t\t{name: "rfc1918", url: "http://10.1.2.3", wantSubstring: "private", wantStage: "blocked request target"},
\t\t{name: "link-local", url: "http://169.254.10.20", wantSubstring: "link-local", wantStage: "blocked request target"},
\t\t{name: "private-dns", url: "http://internal.test/private", wantSubstring: "private", wantStage: "blocked socket target"},
\t}
'''
if text.count(old) != 1:
    raise SystemExit(f"unexpected private-destination table count: {text.count(old)}")
text = text.replace(old, new, 1)

old = '''\t\t\tif !strings.Contains(gotErr, "blocked request target") {
\t\t\t\tt.Fatalf("expected blocked request-target diagnostic, got %q", err.Error())
\t\t\t}
'''
new = '''\t\t\tif !strings.Contains(gotErr, tt.wantStage) {
\t\t\t\tt.Fatalf("expected %q diagnostic, got %q", tt.wantStage, err.Error())
\t\t\t}
'''
if text.count(old) != 1:
    raise SystemExit(f"unexpected private-destination stage assertion count: {text.count(old)}")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
