# JapanFold API

Programmatic access to structure prediction (Boltz-2, ESMFold2, Protenix) and
binder design (BoltzGen), running on Tenstorrent accelerators. The API is an
**async job service**: submit a job, poll its status, then download structures
and scores. Base URL in production is `https://japanfold.com`; every deployment
also exposes the same API locally (e.g. `http://localhost:8080` on a dev box or
an on-prem server).

- Interactive contract: `GET /v1/openapi.json` (OpenAPI 3.1 — feed it to an SDK
  generator or an OpenAPI→MCP tool).
- Auth: an API key sent as `Authorization: Bearer jf_live_...` (or `X-API-Key`).
- Errors: RFC 9457 `application/problem+json` (`type`, `title`, `status`, `detail`).

## Authentication

**None required for the public demo.** Anyone can call `/v1` with no key; requests
run under the same input caps and per-IP rate limits as the web app (see
[Limits](#limits--rate-limiting)), scoped to the caller's IP. This is the
zero-onboarding path for agents (Claude Science, etc.) to try the API.

An **optional** API key raises those limits. Send it as `Authorization: Bearer <key>`
(or `X-API-Key`). A *present but invalid* key is rejected with `401` — omit the
header entirely for the public tier.

Operators mint keys server-side:

```bash
tt-bio apikey create --customer acme --name "prod key"
# -> key: jf_live_...   (shown once; only its SHA-256 is stored)
tt-bio apikey list
```

Keys live in `~/.aiand-bio/api_keys.json` (override with `$JAPANFOLD_API_KEYS_FILE`),
or pass ephemeral keys via `$JAPANFOLD_API_KEYS="customer:key,..."`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | Liveness + API version (public). |
| GET | `/v1/models` | Models, design protocols, parameters, limits (public). |
| POST | `/v1/predictions` | Submit a structure/affinity prediction → **202** job. |
| POST | `/v1/designs` | Submit a BoltzGen binder design → **202** job. |
| GET | `/v1/jobs` | List your jobs (`?limit=&cursor=`). |
| GET | `/v1/jobs/{id}` | Poll one job's status. |
| GET | `/v1/jobs/{id}/results` | Artifact manifest + scores once ready. |
| GET | `/v1/jobs/{id}/artifacts/{path}` | Download one structure/score file. |
| GET | `/v1/jobs/{id}/archive` | Download all results as a zip. |
| GET | `/v1/jobs/{id}/logs` | Plain-text run log. |
| POST | `/v1/jobs/{id}/cancel` | Cancel a running/queued job. |
| DELETE | `/v1/jobs/{id}` | Delete a job and its data. |

Jobs are isolated per key: a key can only ever see its own jobs (others 404).

### Submit a prediction

```bash
curl -sX POST https://japanfold.com/v1/predictions \
  -H "Authorization: Bearer $JAPANFOLD_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: run-42" \
  -d '{"model":"boltz2","name":"mytarget","sequence":"MKTAYIAKQR..."}'
```

Input may be `sequence` (one chain), `input` (a FASTA/YAML string), or `targets`
(a list of `{content, name?}`). `params` accepts `use_msa_server`, `fast`,
`recycling_steps`, `sampling_steps`, `diffusion_samples`, `output_format`.
Send an `Idempotency-Key` to make retried submits return the same job.

The response is a **job**:

```json
{"object":"job","id":"…","kind":"predict","status":"queued","model":"boltz2",
 "progress":null,"links":{"self":"/v1/jobs/…","results":"…/results","archive":"…/archive"}}
```

### Poll → download

```bash
curl -s https://japanfold.com/v1/jobs/$ID -H "Authorization: Bearer $KEY"      # status
curl -s https://japanfold.com/v1/jobs/$ID/results -H "Authorization: Bearer $KEY"  # manifest
curl -sOJ https://japanfold.com/v1/jobs/$ID/archive -H "Authorization: Bearer $KEY" # bundle
```

`status` transitions `queued → running → succeeded | failed | canceled`
(terminal states are immutable). `results` lists downloadable artifacts (CIF/PDB
structures) plus per-target scores (confidence / pLDDT / affinity for
predictions; ranked metrics for designs).

### Synchronous wait (`Prefer: wait`)

To collapse submit-and-poll into one round-trip for fast jobs, send
`Prefer: wait` (or `Prefer: wait=<seconds>`, capped at 60) on the create or the
`GET /v1/jobs/{id}` request. The server holds the response until the job reaches
a terminal state or the wait elapses, then returns the job — `200` if terminal,
`202` if still running (keep polling). This is the Replicate/fal pattern; polling
remains the reliable foundation, and `wait` is a latency optimization.

### Idempotency

Send `Idempotency-Key: <unique>` on a create; a retry with the same key returns
the original job (`200`) instead of launching a duplicate — essential for agents,
which retry on dropped connections.

## Using the CLI

Most users (and agents) drive the API through the `japanfold` CLI, which turns
the submit→poll→download loop into one command:

```bash
japanfold predict --sequence MKTAYIAK... --model boltz2 --wait --out ./out
japanfold design spec.yaml --protocol nanobody-anything --num-designs 10 --wait --out ./out
```

See [`clients/README.md`](../clients/README.md).

## Limits & rate limiting

Numeric parameters are clamped to the deployment's limits (see `/v1/models`).
Per-customer concurrency and submit-rate caps apply; when at capacity the API
returns `429` with a `Retry-After` header. Oversized bodies return `413`.
