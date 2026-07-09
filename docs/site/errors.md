# Errors

Errors are returned as **RFC 9457 problem+json**. The body looks like:

```json
{
  "type":   "https://japanfold.com/errors/invalid-input",
  "title":  "Invalid request",
  "status": 400,
  "detail": "unknown model 'nope' — choose one of ['boltz2', 'esmfold2', 'esmfold2-fast', 'protenix-v2'].",
  "instance": "/v1/predictions"
}
```

- `title` — short, human-readable summary.
- `detail` — specific explanation of *this* failure (read this first).
- `status` — the HTTP status, mirrored in the body.
- `type` — a URI categorizing the error (may be `about:blank`).
- `instance` — the request path that failed.

## Status codes

| Code | Meaning | What to do |
|---|---|---|
| `400` | Invalid request — bad params, malformed input, or over a size cap (residues, chains, designs, …). | Fix the request; read `detail`. See [Models & limits](models-and-limits.md). |
| `401` | Missing/invalid credentials for an authenticated action. | Check your `Authorization: Bearer` key. |
| `403` | Forbidden. | See the Cloudflare note below if `detail` mentions error `1010`. |
| `404` | No such job — or a job you don't own. | Check the id; jobs are scoped to their owner. |
| `413` | Request body over 8 MB. | Shrink the input — you're almost certainly over `max_content_chars` anyway (see [Models & limits](models-and-limits.md)). |
| `429` | At capacity or over a rate limit. | **Honor the `Retry-After` header**, then retry. See the limits page. |

## Handling 429 (at capacity)

The service is a free demo on shared compute. When it is busy — or you exceed a
submit/active-job quota — you get `429` with a `Retry-After` header (seconds).
Wait that long and retry; back off if it repeats.

```bash
resp=$(curl -s -D /tmp/h -o /tmp/b -w '%{http_code}' -X POST \
  https://api.japanfold.com/v1/predictions -H 'Content-Type: application/json' \
  -d '{"model":"boltz2","sequence":"MKTAYIAK..."}')
if [ "$resp" = "429" ]; then
  sleep "$(grep -i '^retry-after:' /tmp/h | tr -dc 0-9)"
  # retry...
fi
```

## Cloudflare 403 / error 1010

A `403` whose body references Cloudflare **error 1010** is edge-level bot
filtering of your HTTP client, not an API error. Retry the identical request with
a browser-like `User-Agent`:

```bash
curl -s -A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36' \
  https://api.japanfold.com/v1/models
```
