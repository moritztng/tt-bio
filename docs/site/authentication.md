# Authentication

The API is **public and keyless**. Every endpoint works with no credentials —
just send the request. This is the same access the web app at
[japanfold.com](https://japanfold.com) uses, with the same limits.

```bash
# No auth needed:
curl -s https://api.japanfold.com/v1/models
```

## Optional API key — raised limits

Send a key of the form `jf_live_…` to lift the free-tier caps (larger
structures, more chains, more jobs at once) — as `Authorization: Bearer` or
`X-API-Key`:

```bash
curl -s https://api.japanfold.com/v1/predictions \
  -H 'Authorization: Bearer jf_live_your_key_here' \
  -H 'Content-Type: application/json' \
  -d '{"model":"boltz2","sequence":"MKTAYIAK..."}'
```

A key that's present but invalid is rejected with `401` — omit the header
entirely to fall back to the public tier. Keys are issued by the JapanFold
team; there's no self-serve signup yet.

## Ownership and quotas

- **Keyless requests** are grouped and rate-limited per client IP (and per
  session for the web app). This is a free demo on shared compute, so there are
  quotas on how much you can run at once.
- **Keyed requests** are owned by your key. You can only list, poll, cancel or
  delete jobs that belong to you (your IP/session for keyless use, your key
  otherwise). Accessing someone else's job returns `404`.
- The concrete numbers — active-job quotas, submit rates, structure/design size
  caps — are in **[Models & limits](models-and-limits.md)** and, live, at
  `GET /v1/models`.

Over a cap you get `400`; at capacity you get `429` with a `Retry-After` header
(see [Errors](errors.md)).
