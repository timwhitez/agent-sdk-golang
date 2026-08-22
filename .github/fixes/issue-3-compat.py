from pathlib import Path

path = Path("sdk/tools/sandbox/sandbox_webfetch.go")
text = path.read_text()
old = '''func newWebfetchHTTPClient(timeout time.Duration) *http.Client {
\tvar transport *http.Transport
\tif base, ok := http.DefaultTransport.(*http.Transport); ok && base != nil {
\t\ttransport = base.Clone()
\t} else {
\t\ttransport = &http.Transport{ForceAttemptHTTP2: true}
\t}
\t// A proxy would resolve the target independently and defeat destination
\t// pinning. Webfetch therefore connects directly to the validated address.
\ttransport.Proxy = nil
\ttransport.DialContext = dialValidatedWebfetchDestination
\treturn &http.Client{Timeout: timeout, Transport: transport}
}
'''
new = '''func newWebfetchHTTPClient(timeout time.Duration) *http.Client {
\tif base, ok := http.DefaultTransport.(*http.Transport); ok && base != nil {
\t\ttransport := base.Clone()
\t\t// A proxy would resolve the target independently and defeat destination
\t\t// pinning. Webfetch therefore connects directly to the validated address.
\t\ttransport.Proxy = nil
\t\ttransport.DialContext = dialValidatedWebfetchDestination
\t\treturn &http.Client{Timeout: timeout, Transport: transport}
\t}
\t// A non-*http.Transport value is an explicit host/test replacement that may
\t// implement its own connection policy. Preserve that injection seam instead
\t// of silently bypassing it with a new default transport.
\treturn &http.Client{Timeout: timeout, Transport: http.DefaultTransport}
}
'''
if text.count(old) != 1:
    raise SystemExit(f"compat transport anchor count={text.count(old)}")
path.write_text(text.replace(old, new))
