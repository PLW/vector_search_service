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

### 0) High-level shape

Each **shard** owns a stream of **immutable segments** (LSM-ish), plus a **tombstone overlay** to guarantee deletes in ≤4s. Queries run against a *stable snapshot* (manifest version) + live tombstone set; compaction builds new segments off to the side and atomically swaps manifests (no query blocking).

---

## 1) Segment file layout (on object store or local NVMe + replicated)

A segment is a directory/prefix `seg_<segid>/` containing:

1. **`segment.meta`** (small protobuf/flatbuffer, checksummed)

* `segid`, `tenant_id` (or “mixed-tenant” flag if you pack), `shard_id`
* `build_time`, `format_version`
* `vector_dim=768`, `metric` (cos/dot)
* `n_points_total`
* `docid_min/max` (internal)
* `tag_schema_hash` (allowlisted keys & enum maps version)
* Bloom params / filter stats

2. **`docstore.bin`** (point records, append-only)

* Fixed record header: `docid u32`, `id_hash u64`, `version u64`, `vector_off u64`, `tags_off u64`
* Optional: `external_id` stored in `id_table` below (docstore keeps only hash + docid)
* This is the authoritative mapping for docid→(id/version/vector/tags)

3. **`vectors.f16`** (dense)

* Contiguous float16 array: `[n_points][768]` aligned (e.g., 64B)
* Optional per-segment quantization artifacts (PQ codes) if used.

4. **ANN index files** (choose one family; layout below supports both)

* **IVF-PQ path**

  * `coarse_centroids.f32` (Kc x 768)
  * `ivf_lists.idx` (offset table per list)
  * `ivf_lists.codes` (PQ codes per vector in posting order)
  * `ivf_lists.docids` (docid per code, parallel array)

* **HNSW path**

  * `hnsw.graph` (levels + adjacency lists)
  * `hnsw.entrypoints` (per level)
  * `hnsw.docids` (node→docid)

5. **Tag filter index**

* Tags are low-cardinality; store per-key dictionary and per-value postings:

  * `tags.dict` (key→value dictionary, value→value_id)
  * `tags.postings` (value_id → roaring bitmap of docids) OR (value_id → sorted docid list + skip)
  * `tags.stats` (counts)
* Keep this *segment-local* and composable (AND across keys done by intersecting bitmaps/lists).

6. **`id_table`** (optional, for fast external-id→docid lookups inside segment)

* `id_hash -> docid` minimal perfect hash or sorted table + binary search
* If you need exact external IDs, also store `external_id` string table:

  * `ids.strings` (concatenated) + `ids.offsets`

7. **`checksums`** / `manifest_ref`

* Per-file checksums; a final “sealed” marker `SEALED` written last.

**Segment invariants**

* Immutable after `SEALED`.
* Contains **only latest-known versions at build time** (compaction enforces), but queries still must consult tombstones (and optional version overlay).

---

## 2) In-memory structures (query-serving on each shard replica)

### 2.1 Manifest snapshot

* **`Manifest`**: current active set of segments for shard (and per tenant, if partitioned).

  * `active_segments: [segid...]` ordered by “recency” / level
  * Per segment: memory-mapped handles for ANN + tags + docstore indexes
* Query uses `(manifest_version, tombstone_epoch)` as a consistent read view.

### 2.2 Tombstone overlay (hard delete ≤4s)

Two-tier design to meet the 4s rule without touching segments:

1. **Hot tombstone set (RAM, replicated)**

* Key: `(tenant_id, external_id_hash)` or `(tenant_id, internal_point_key)`
* Value: `delete_version`, `delete_time`, optional `reason`
* Data structure: sharded concurrent hash set + time wheel / min-heap for expiry.
* This is what queries check synchronously.

2. **Warm tombstone log (durable)**

* Append-only log per shard (or per tenant partition) persisted to WAL / replicated log.
* Replayed on restart to rebuild hot set quickly.
* Periodically compacted to a checkpoint file `tombstones.chkpt`.

**Lookup on query**

* After ANN returns candidate docids, resolve to `id_hash/version` via docstore index, then:

  * If `tombstone.contains(tenant_id, id_hash)` with `delete_version >= candidate_version` → drop.
* (Optional) also consult `upsert_version_map` if you choose to support strict last-write-wins at read time.

### 2.3 Upsert version overlay (optional but recommended)

To avoid returning stale older versions until compaction catches up:

* `LatestVersionMap`: `(tenant_id,id_hash)->latest_version` (and maybe `latest_doc_pointer` if you keep an in-memory “delta segment”).
* During search, drop candidates whose `candidate_version < latest_version_map[...]`.
* This can be bounded in size with per-tenant quotas and periodic snapshotting.

---

## 3) Manifest format (atomic swaps; rebuilds never block queries)

A shard has a **manifest log** plus periodic checkpoints.

### 3.1 `manifest.chkpt` (protobuf/flatbuffer)

* `manifest_version`
* `segments_active`: list of `segid`, `level`, `min/max id_hash` (optional), `n_points`, `build_time`
* `segments_inflight`: (optional) segids being built; **not query-visible**
* `tag_schema_version`
* `tombstone_checkpoint_ref` (for fast bootstrap)
* `compaction_policy_state` (optional stats)

### 3.2 `manifest.log` (append-only)

Records:

* `ADD_SEGMENT(segid, level, file_refs, stats)`
* `REMOVE_SEGMENT(segid)`
* `PROMOTE_SEGMENT(segid, new_level)` (optional)
* `SET_SCHEMA(tag_schema_version)`
* `SET_TOMBSTONE_CHECKPOINT(ref)`

**Atomicity**

* Build new segment → write files → write `SEALED` → then append `ADD_SEGMENT` to manifest log.
* Compaction swap = append `ADD_SEGMENT(newseg)` then `REMOVE_SEGMENT(oldsegs...)` in a single “transaction record” (or bracket with `BEGIN/COMMIT`).

**Query non-blocking**

* Query threads hold an `Arc<ManifestSnapshot>` pointer; swaps are atomic pointer flips.
* Old snapshots remain valid until refcount drops; segments are garbage collected later.

---

## 4) Write path: inserts/upserts/deletes

### 4.1 Upsert / insert

* Append to **memtable / delta segment builder**:

  * Either a small in-memory HNSW (fast ingest) or buffered vectors + periodic flush to immutable segment.
* Update `LatestVersionMap` for `(tenant,id_hash)` with new version.
* When flush triggers, create **Level-0** segment:

  * Contains new vectors + tags + docstore + ANN structure.
  * Sealed and added to manifest.

### 4.2 Delete

* Append to tombstone WAL; replicate.
* Apply to Hot tombstone set immediately (on ack path).
* Also update `LatestVersionMap` if you treat delete as a versioned write.
* Compaction will eventually drop deleted entries physically.

**Delete visibility guarantee**

* Query nodes must subscribe to tombstone replication stream; enforce “ack after quorum applied on query-serving replicas”.
* Target: 4s worst-case including replication + apply.

---

## 5) Compaction triggers & policy

Think LSM with levels:

* **L0**: many small fresh segments (fast flush, high overlap).
* **L1/L2/...**: fewer larger segments, disjoint by key range (or hashed partitions).

### 5.1 Triggers

1. **L0 count / size**

* If `L0_segments > N` or `L0_bytes > B`, schedule compaction of a batch.

2. **Overlap / duplicate pressure**

* If `duplicate_rate` (older versions) estimated high (via LatestVersionMap sampling), compact sooner.

3. **Tombstone pressure**

* If `tombstone_hit_rate` in queries > threshold OR tombstone set size > threshold → compact to reclaim.

4. **Read amplification / latency regression**

* If p95 query time rises due to too many segments, compact.

5. **Time-based**

* Periodic compaction window to smooth tail latency.

### 5.2 What compaction does

Given input segments S1..Sk:

* Merge docstores by `(tenant_id,id_hash)` keeping only max `version` not tombstoned.
* Rebuild tag postings for surviving docids.
* Rebuild ANN index for the output segment.
* Emit new segment `Sout` at higher level; atomically manifest swap.

**Important:** compaction does *not* need to block queries; it reads old segments via mmap and writes new files elsewhere.

---

## 6) Segment lifecycle state machine

Per segment `segid`, maintained by the shard’s segment manager.

### States

1. **`ALLOCATED`**

* segid reserved; directory/prefix created; not visible.

2. **`BUILDING`**

* Writing vectors/docstore/index/tag postings.
* Segment is *not* queryable.

3. **`SEALED`**

* All files written, checksummed, `SEALED` marker present.
* Still not queryable until manifest add.

4. **`PUBLISHED`**

* `ADD_SEGMENT` committed to manifest; segment becomes query-visible in new snapshots.

5. **`DEPRECATED`**

* Segment has been superseded by compaction (or schema migration).
* It may still be referenced by older manifest snapshots; must remain readable.

6. **`RECLAIMABLE`**

* No live manifest snapshot references it (refcount==0) AND retention grace period passed.

7. **`DELETING`**

* Physical deletion of files/prefix in progress (or mark-for-delete in object store).

8. **`DELETED`**

* Files removed; terminal.

### Transitions (events)

* `ALLOCATED -> BUILDING` : builder starts
* `BUILDING -> SEALED` : finalize + checksums + write `SEALED`
* `SEALED -> PUBLISHED` : manifest transaction commits `ADD_SEGMENT`
* `PUBLISHED -> DEPRECATED` : manifest commits swap removing it (compaction) or `REMOVE_SEGMENT`
* `DEPRECATED -> RECLAIMABLE` : segment manager observes no snapshot refs
* `RECLAIMABLE -> DELETING` : GC worker starts deletion
* `DELETING -> DELETED` : delete complete

### Failure handling (key non-blocking rules)

* Crash in `BUILDING`: segment ignored (no `SEALED`), safe to GC.
* Crash after `SEALED` but before `PUBLISHED`: safe; recovery can either publish (if intended) or GC if orphaned (no manifest reference).
* Crash during manifest swap: use manifest log transaction semantics (BEGIN/COMMIT) to ensure atomic visibility.
* Queries always operate on last committed manifest snapshot; rebuilds cannot partially surface.

---

## 7) Notes on “rebuilds shouldn’t block queries”

* “Rebuild” (e.g., ANN parameter change or schema change) is implemented as **shadow build**:

  * Build new segments in parallel, publish alongside old under a new “index epoch”, then flip routing once warm.
* Queries hold snapshot pointers; no global locks.
* If you need per-tenant rebuild, you can flip only that tenant’s routing table.

