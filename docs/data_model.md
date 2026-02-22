# Distributed Vector Store
## Index storage format: immutable segments, background compaction

Generated from initial ChatGPT prompt:
```
Define the index storage format as immutable segments + background compaction.
Requirements:
  - Upserts and deletes supported
  - Deletes visible in <=4s
  - Rebuilds should not block queries

Specify: segment file layout, in-memory structures, tombstones,
  manifest, and compaction triggers.

Layout format: all data items in each file should be specified as
  TYPE: NAME, NAME, ..
  For example: `u64 segid, tenant_id, shard_id`

Provide a state machine for “segment lifecycle”.
```

## Immutable-segment index format + background compaction

### Design goals

* **Immutable segments** (LSM-style) for safe concurrent reads and cheap atomic publishes.
* **Deletes visible ≤ 4s** via a replicated tombstone overlay that queries must consult.
* **Rebuilds/compactions never block queries**: build new segments off to the side, then atomically swap manifests; queries use snapshot pointers.

---

## 1) Segment file layout (per shard, per segment `seg_<segid>/`)

All files are written, checksummed, then a `SEALED` marker is created last.

### Core metadata

* **`segment.meta`** (small, checksummed; protobuf/flatbuffer)

  * `segid`, `shard_id`, `tenant_partition` (or “mixed-tenant” flag)
  * `format_version`
  * `dim=768`, `metric` (cos/dot), `vector_dtype=f16`
  * `n_points`
  * `build_time`, `min_version/max_version`
  * `tag_schema_version/hash` (allowlisted keys + enum maps)
  * stats: tag cardinalities, IVF/HNSW params, CRCs, bloom params

### Point store (authoritative mapping)

* **`docstore.rec`**

  * Record per point (fixed or var):

    * `docid u32` (segment-local)
    * `id_hash u64` (hash of external id; optional external-id reference)
    * `version u64`
    * `vector_offset u64` (into vectors file)
    * `tags_offset u64`
* **`ids.str` + `ids.off`** (optional, if you need exact external IDs, not just hash)
* **`docstore.idx`** (optional accel): `id_hash -> docid` (sorted table or mphf)

### Vectors

* **`vectors.f16`**: contiguous `[n_points][768]` float16, aligned (64B+).

### ANN index (choose one family per index)

**IVF-PQ variant**

* `coarse_centroids.f32`
* `ivf.list_off` (K lists offsets)
* `ivf.docids` (docids in list order)
* `ivf.codes` (PQ codes aligned with docids)
* (optional) `residuals.f16` or `opq.mat` if used

**HNSW variant**

* `hnsw.levels` / `hnsw.graph` (adjacency lists)
* `hnsw.entry` (entry points)
* `hnsw.node_docid` (node -> docid)

### Tag filter index (small set of tags, low-ish cardinality)

* **`tags.dict`**: per-key value dictionary (`value -> value_id`)
* **`tags.postings`**: `value_id -> roaring bitmap(docid)` (or sorted lists + skip)
* **`tags.stats`**: counts to short-circuit segments with no matches

### Integrity / sealing

* **`checksums`**: per-file CRC/xxhash
* **`SEALED`**: empty file written last (commit marker)

**Segment invariants**

* Immutable after `SEALED`.
* Segment may contain older versions (esp. L0); read path must enforce last-write-wins and deletes via overlays until compaction cleans it up.

---

## 2) In-memory structures (per shard replica)

### 2.1 Manifest snapshot pointer

* `Atomic<Arc<ManifestSnapshot>> current_manifest`
* Snapshot contains:

  * `manifest_version`
  * ordered `active_segments[]` with mmap handles + per-segment cached headers
  * `level`/tier metadata
* Queries grab a snapshot pointer and never block on swaps.

### 2.2 In-memory “delta” (mutable) segment for fresh writes

* **Memtable / DeltaIndex**

  * `id_hash -> (version, doc_record_ptr)`
  * vector store for newly ingested points (float16 or float32 in RAM)
  * small ANN structure (e.g., HNSW-in-RAM) OR brute-force block scan for small N
  * tag postings maintained incrementally (bitsets/roaring)
* Periodically flushed into an immutable **L0 segment**.

### 2.3 Tombstones (to guarantee delete ≤4s)

Two-tier tombstone system:

1. **Hot Tombstone Map (RAM)**

* Key: `(tenant_id, id_hash)`
* Value: `delete_version`, `delete_time`
* Concurrent hash map + time-index for retention/eviction.
* Queried synchronously on every candidate before returning.

2. **Durable Tombstone Log / WAL**

* Replicated append-only log (per shard) of deletes (and optionally upserts).
* Periodic checkpoint `tombstone.chkpt` to bound replay time.

**Delete visibility contract**

* Delete is *acknowledged* only after:

  * quorum commit in replicated log **and**
  * applied to hot tombstone map on all query-serving replicas for that shard (or routing ensures only caught-up replicas serve queries).
* This is what makes “≤4s” enforceable.

### 2.4 Latest-version overlay (recommended)

* `LatestVersionMap: (tenant_id,id_hash) -> latest_version`
* Used at query-time: discard candidates whose `candidate.version < latest_version`.
* Keeps correctness (last-write-wins) even when older segments still contain stale versions.

---

## 3) Manifest (atomic publish / non-blocking swaps)

### 3.1 Structures

* **`manifest.chkpt`** (full state snapshot)

  * `manifest_version`
  * `active_segments`: `(segid, level, stats, file_refs)`
  * `tag_schema_version`
  * pointers to tombstone checkpoint, WAL truncation point
* **`manifest.log`** (append-only)

  * `ADD_SEGMENT(segid, level, stats, refs)`
  * `REMOVE_SEGMENT(segid)`
  * `SET_SCHEMA(version)`
  * `SET_TOMBSTONE_CHECKPOINT(ref)`
  * (optional) `BEGIN_TXN/COMMIT_TXN` for atomic multi-record swaps

### 3.2 Publish protocol (never blocks queries)

1. Build segment off to the side → write files → `checksums` → `SEALED`
2. Append `ADD_SEGMENT` to manifest log (durable/replicated)
3. Atomically swap `current_manifest` pointer to new snapshot
4. Old snapshot remains valid until refcount drops

Rebuilds follow the same: build new segments in parallel, then manifest flip.

---

## 4) Compaction triggers (LSM policy)

Segments are tiered: **L0 (fresh, overlapping)** → **L1/L2… (larger, fewer)**.

Trigger compaction when any threshold is exceeded:

1. **L0 count/bytes**

* `L0_segments > N0` or `L0_bytes > B0` (controls read amplification)

2. **Duplicate/version pressure**

* sample-based estimate of stale versions high (many candidates dropped by LatestVersionMap)

3. **Tombstone pressure**

* hot tombstone map size grows beyond threshold
* high tombstone-hit-rate observed in queries (wasted work)

4. **Latency regression**

* p95 query cost rises due to segment fanout; compact to reduce segment count

5. **Age-based smoothing**

* periodic compaction to avoid bursty large merges

### Compaction operation

Input: a set of segments `{S1..Sk}`.

* Merge by `(tenant_id,id_hash)` keeping only max version not tombstoned.
* Rebuild tag postings on survivors.
* Rebuild ANN index (IVF/HNSW) for output `Sout` (higher level).
* Publish: `ADD_SEGMENT(Sout)` + `REMOVE_SEGMENT(S1..Sk)` atomically in manifest log.
* Physical deletion of old segments deferred until no snapshots reference them.

---

## 5) Segment lifecycle state machine

### States

1. **ALLOCATED**

* segid reserved, location chosen; not visible.

2. **BUILDING**

* writing vectors/docstore/tags/ANN; may crash safely.

3. **SEALED**

* all files complete + checksummed + `SEALED` marker present; still not query-visible.

4. **PUBLISHED**

* referenced by a committed manifest snapshot; query-visible.

5. **DEPRECATED**

* removed from latest manifest by compaction/rebuild; may still be referenced by older snapshots.

6. **RECLAIMABLE**

* not referenced by any live snapshot (refcount==0) and retention grace passed.

7. **DELETING**

* GC worker deleting underlying files/prefix.

8. **DELETED**

* terminal.

### Transitions

* `ALLOCATED -> BUILDING` : segment builder starts
* `BUILDING -> SEALED` : finalize + write `SEALED`
* `SEALED -> PUBLISHED` : manifest commits `ADD_SEGMENT`
* `PUBLISHED -> DEPRECATED` : manifest commits swap removing it
* `DEPRECATED -> RECLAIMABLE` : no snapshot references remain
* `RECLAIMABLE -> DELETING` : GC begins
* `DELETING -> DELETED` : deletion complete

### Failure rules (key to “rebuilds don’t block queries”)

* Crash in `BUILDING`: ignore (no `SEALED`), GC later.
* Crash after `SEALED` but before `PUBLISHED`: orphan; recovery can GC or publish if referenced by manifest intent.
* Manifest swap uses transactional log records so queries only ever see fully committed segment sets.
