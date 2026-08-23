from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_test.go")
text = path.read_text(encoding="utf-8")
old = '''\torigTransport := http.DefaultTransport
\tvar calls int
\thttp.DefaultTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
\t\tcalls++
\t\tif calls > 1 {
\t\t\tt.Fatalf("redirect request should not run for blocked private destination")
\t\t}
\t\tresp := &http.Response{
\t\t\tStatus:     "302 Found",
\t\t\tStatusCode: http.StatusFound,
\t\t\tHeader:     make(http.Header),
\t\t\tBody:       io.NopCloser(strings.NewReader("redirect")),
\t\t\tRequest:    r,
\t\t}
\t\tresp.Header.Set("Location", "http://internal.test/private")
\t\treturn resp, nil
\t})
\tt.Cleanup(func() { http.DefaultTransport = origTransport })
'''
new = '''\torigDo := webfetchDoRequest
\tvar calls int
\twebfetchDoRequest = func(client *http.Client, r *http.Request) (*http.Response, error) {
\t\tcalls++
\t\tif calls > 1 {
\t\t\tt.Fatalf("redirect request should not run for blocked private destination")
\t\t}
\t\tredirectRequest, requestErr := http.NewRequestWithContext(r.Context(), http.MethodGet, "http://internal.test/private", nil)
\t\tif requestErr != nil {
\t\t\treturn nil, requestErr
\t\t}
\t\tif client.CheckRedirect == nil {
\t\t\treturn nil, errors.New("WebFetch client has no redirect policy")
\t\t}
\t\tif redirectErr := client.CheckRedirect(redirectRequest, []*http.Request{r}); redirectErr != nil {
\t\t\treturn nil, redirectErr
\t\t}
\t\treturn nil, errors.New("private redirect was unexpectedly accepted")
\t}
\tt.Cleanup(func() { webfetchDoRequest = origDo })
'''
if text.count(old) != 1:
    raise SystemExit(f"unexpected private redirect fixture count: {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
