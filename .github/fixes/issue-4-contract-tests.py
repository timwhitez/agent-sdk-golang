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
path.write_text(text.replace(old, new, 1), encoding="utf-8")
