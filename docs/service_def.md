## Distributed Vector Search Service — One-page Service Contract
(Note: ChapGPT initial generator)

### 1) Purpose

Provide low-latency approximate nearest-neighbor (ANN) search over **1B 768-dim vectors** with **metadata tag filters** and **multi-tenant isolation**, supporting **insert/upsert/delete**. Reads may be slightly stale, but **deletes must stop matching within 4s**.

---

### 2) Service Level Objectives (SLOs)

**Latency (topK=20):**

* p95 **< 50ms** end-to-end at steady **2k QPS**, burst **5k QPS** (with adaptive load shedding below).
* p99 target: **< 120ms** (best-effort; not a hard SLO).

**Availability:**

* Monthly read availability: **99.9%** (success = non-error response, even if partial per semantics).
* Write availability: **99.5%** (insert/upsert/delete accepted).

**Freshness / Staleness:**

* **Index visibility lag (upserts/inserts):** bounded; target **< 30s** to be searchable.
* **Delete visibility lag:** hard requirement **≤ 4s** from accepted delete to “will not be returned”.

**Correctness (ANN nature):**

* Recall is **best-effort** and tunable; no guaranteed exact nearest neighbors.
* Filter correctness: tag filters must be applied correctly for returned results.

**Multi-tenant protection:**

* Per-tenant quotas enforced; noisy neighbors must not violate latency SLOs for other tenants under provisioned capacity.

---

### 3) Data Model

**Identity & tenancy**

* `tenant_id` (required), `id` (unique within tenant)

**Vector**

* `vector`: float16 (storage), query accepts float16/float32; service normalizes to internal format.
* `dim` fixed at **768** per index.

**Metadata**

* `tags`: small set of low-cardinality attributes (e.g., `{"lang":"en","doc_type":"pdf","region":"us"}`), with **allowlisted keys per tenant/index**.
* Optional numeric fields for rerank/hybrid (see below) only if explicitly enabled.

**Versioning**

* Each point has a monotonic `version` (client-supplied or server-assigned). Upserts are last-write-wins by `(tenant_id, id)`.

---

### 4) APIs (HTTP/gRPC; semantics identical)

#### 4.1 Search

`POST /v1/search`

```json
{
  "tenant_id": "t1",
  "index": "main",
  "query_vector": [ ... ],
  "top_k": 20,
  "filter": { "tags": { "lang": ["en"], "doc_type": ["pdf"] } },
  "hybrid": { "text_query": "optional", "alpha": 0.3 },
  "consistency": { "max_staleness_ms": 30000 }
}
```

**Response**

```json
{
  "results": [
    { "id":"p123", "score":0.812, "tags":{...}, "version":42 },
    { "id":"p456", "score":0.812, "tags":{...}, "version":42 },
    ...
  ],
  "took_ms": 18,
  "partial": false
}
```

**Semantics**

* **Best-effort ANN**: returns approximate topK by `score` (vector similarity unless hybrid enabled).
* **Filters**: applied as conjunctive constraints over allowlisted tag keys.
* **Staleness**: service may use an index snapshot up to `max_staleness_ms` old (server may clamp to configured bounds).
* **Delete guarantee**: results must not include items deleted >4s ago (see §6).

**Errors**

* `400` Bad Request - bad dim / invalid filters
* `429` Too Many Requests - tenant throttled (quota limit)
* `501` Not Implemented - v1 operations only
* `503` Service Unavailable - overload (after throttling) / maintenance 

#### 4.2 Upsert / Insert

`POST /v1/upsert

```json
{
  "tenant_id":"t1",
  "index":"main",
  "vectors":[
    {"id":"p123","vector":[...],"tags":{"lang":"en"}, "version":42},
    {"id":"p456","vector":[...],"tags":{"lang":"en"}, "version":42},
    ...
  ]
}
```

**Semantics**

* Upsert is idempotent by `(tenant_id,id,version)` if version provided.
* Visibility to search is **eventually consistent**, target <30s (not guaranteed).

#### 4.3 Delete

`POST /v1/delete`

```json
{ "tenant_id":"t1","index":"main","ids":["p123","p124"], "version":43 }
```

**Semantics**

* Delete is idempotent; last-write-wins with versioning.
* **Hard propagation SLO:** within **4s** of accepted delete, the point **must not appear** in search results (even if index snapshots are stale).

#### 4.4 Management / Quotas / Health

* `GET /v1/tenants/{tenant_id}/quota` (limits & current usage)
* `GET /v1/indexes/{index}/stats` (size, shard map, build lag, delete-lag)
* `GET /healthz` (liveness), `GET /readyz` (readiness)

---

### 5) Scoring & Filtering Semantics

**Vector similarity**

* Default: cosine or dot-product (index-configurable per index; fixed after creation).
* Input vectors may be normalized server-side depending on metric.

**Metadata filters**

* Supported operators (v1):

  * `tags.key IN [v1, v2, ...]`
  * AND across keys
* Not supported (v1): arbitrary boolean logic, range queries (unless explicitly enabled as “numeric filters v2”).

**Hybrid scoring (optional)**

* If enabled, combines vector score with text/BM25 score:

  * `score = (1-α)*vector_score + α*text_score`
* Text index is optional per index; absent => request rejected if `hybrid` specified.

---

### 6) Consistency & Delete Enforcement (explicit contract)

**Read staleness**

* Searches are served from a **shard-local index snapshot** that may lag writes (bounded by config/`max_staleness_ms`).

**Delete correctness within 4 seconds**

* Deletes are enforced by a **distributed tombstone service** that is checked on every candidate result path:

  * A delete is “effective” when the tombstone is replicated to all query-serving nodes responsible for the tenant’s shards.
  * Query nodes must apply tombstones as a **hard filter** after ANN retrieval and before returning results.
* If tombstone state is unavailable, query nodes must fail closed for affected tenant/shards (return `503` or `partial=true` with zero results from that shard per policy).

---

### 7) Multi-tenant Quotas & Isolation

**Quotas (per tenant)**

* Storage: max vectors, max bytes (vectors + tags), max indexes
* Throughput: max QPS sustained, max burst QPS, max write QPS
* Concurrency: max in-flight requests
* Optional: per-tenant minimum reserved capacity class

**Enforcement**

* Token-bucket per tenant at ingress (global) + local shed at query nodes.
* Fair scheduling across tenants; heavy tenants cannot starve others.

---

### 8) Explicit Non-goals

* **No exact KNN guarantee** or strict recall SLA (ANN system).
* **No transactional semantics** across multiple points; no multi-key atomicity.
* **No strong read-after-write consistency** for upserts/inserts.
* **No complex filter language** (OR/NOT/nested predicates) in v1.
* **No cross-tenant queries**; no joins/aggregations/analytics.
* **No per-request custom distance functions** beyond configured metric.
* **No guarantee of stable ordering** when scores tie (except deterministic per shard best-effort).

