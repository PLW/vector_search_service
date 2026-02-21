# Distributed Vector Search
(Note:  initial generation by ChatGPT LLM.)

## (1) Durability when you have an in-memory segment

You need **write-ahead durability that is independent of the in-memory index**. The in-memory segment is just a serving/ingest optimization; durability comes from a replicated log + checkpoints.

### Write pipeline (per shard)

1. **Ingress → shard leader** (for the relevant partition of `(tenant_id, id_hash)`).
2. Leader appends a **mutation record** to a **replicated WAL** (Raft/Paxos or equivalent):

   * `UPSERT(tenant,id,version,tags,vector)` or `DELETE(tenant,id,version)`
   * record includes a checksum and schema version.
3. Mutation is considered **durable/ackable** only after **quorum commit** (e.g., 2/3) on storage nodes.
4. Only after commit, apply to:

   * **Memtable / in-memory segment builder** (vector + tags)
   * **LatestVersionMap**
   * **Hot Tombstone Set** (for deletes)

This guarantees durability even if the process dies before flush: on restart, you replay the WAL from the last checkpoint.

### Checkpointing & flush

* Periodically (or by size), you create an **immutable L0 segment** from the in-memory builder:

  * build files (`vectors.f16`, ANN, tags postings, docstore), write `SEALED`.
  * then commit `ADD_SEGMENT(segid)` to the **manifest** (also replicated/durable).
* After the segment publish is committed, you can advance the WAL truncation point:

  * write a **checkpoint record**: “all mutations ≤ LSN X are reflected in published segments + tombstone checkpoint”.
  * followers can compact WAL safely.

### Why you still might “lose” without WAL

If you only kept vectors in RAM until flush, a crash loses those points. The WAL is the *only* thing that makes the in-memory segment safe.

### Deletes ≤ 4s guarantee (ties to WAL)

* Deletes go through the same quorum-committed WAL.
* Query-serving replicas subscribe to WAL stream (or a derived tombstone stream) and must apply tombstones immediately upon commit.
* You ack delete when quorum has committed **and** all query-serving replicas for that shard have applied (or you route queries only to replicas that are caught up). This is how you make “≤4s” enforceable.

---

## (2) Querying with large merged segments (and many segments total)

You want two properties:

1. **Small, predictable fanout** at query time.
2. **Don’t scan giant segments unnecessarily**, especially when filters are selective.

The strategy is: **segment-tiering + per-segment candidate generation + global topK merge**, with aggressive pruning.

### 2.1 Segment organization to keep fanout bounded

* L0: small, recent segments (maybe multiple)
* L1+: larger segments produced by compaction
* **Key rule:** keep the number of *active* segments per shard bounded (e.g., 1–2 large L2/L3 segments + a handful of L0/L1).
* Compaction policy is tuned specifically to keep read amplification stable: when segment count crosses threshold, compact.

In practice, a shard should serve from something like:

* 1–3 big segments (multi-GB) at higher levels
* 4–12 small recent segments
  Not hundreds.

### 2.2 How a query runs across segments

Given query vector q, filter F, topK=20:

**Step A — Build a per-query filter plan**

* If `F` is present, compute a **segment-level eligibility**:

  * Using per-segment tag stats or a tag Bloom/dictionary: quickly decide if segment can possibly match.
* For eligible segments, prepare a **docid allowlist bitmap**:

  * intersect tag bitmaps per key/value within that segment.
  * If the filter is selective, you get a small candidate universe fast.

**Step B — Candidate generation per segment**

* For each eligible segment `S`:

  * Run ANN search with a **budget** (efSearch/nprobe tuned by level and remaining latency budget).
  * If filter is selective:

    * either use “filtered ANN” (if your index supports it), or
    * run ANN to get M candidates then apply the allowlist bitmap to keep matches.
  * Apply **tombstone + latest-version** checks before emitting candidates.
  * Emit up to `K'` candidates (e.g., 50–200) with scores.

This is embarrassingly parallel across segments and can be executed with a per-query micro-scheduler.

**Step C — Global topK merge**

* Merge candidates from all segments with a size-K min-heap.
* Optional: do a lightweight **rerank** on the final shortlist using exact dot-product on float16/float32 vectors (common trick; improves recall without increasing ANN work too much).

### 2.3 Handling a *very large merged segment*

A large segment is not inherently a problem if:

* it’s **memory-mapped** (vectors + index) and
* ANN search cost is sublinear and tuned.

But you must avoid “big segment = big cold IO”. Common tactics:

**(a) Memory map + hot working set**

* Keep ANN graph / IVF centroids + coarse structures hot in RAM.
* Vectors can be memory-mapped; rerank reads a small number of vectors (say 200–1000), which is manageable if access is reasonably random but still small.

**(b) Two-stage search in big segments**

* Stage 1: ANN returns `M` docids (fast, mostly CPU).
* Stage 2: exact score reads `M` vectors; choose `M` based on latency budget.

**(c) Partition within “big”**
Even inside a large segment, the ANN index should be internally partitioned:

* IVF: many lists, only probe `nprobe` lists
* HNSW: graph search visits bounded nodes with efSearch

**(d) Filter-driven shortcuts**
If filter allowlist is small, don’t do a full ANN search:

* Option 1: directly score the allowlist (if it’s tiny; e.g., ≤10k points).
* Option 2: maintain per-tag “mini-index” (e.g., tag-specific IVF) for very common filters if you can justify it.

### 2.4 Keeping rebuild/compaction from hurting queries

Large compactions produce new segments “off to the side.” Query impact is controlled by:

* **I/O scheduling**: compaction has low priority, rate-limited bandwidth.
* **CPU budgeting**: compaction threads are capped; query threads preempt.
* **Atomic publish**: only after `SEALED` + manifest commit.
* **Snapshot reads**: queries continue using old segments until they naturally switch to new manifest pointer.

No blocking; at worst you have temporary extra segments until swap.

---

## One concrete mechanism summary

* **Durability:** quorum-committed WAL (mutations) + manifest log + checkpoints; in-memory index is derived state.
* **Large segments at query time:** bounded active segment count + per-segment ANN with budgets + filter allowlists + tombstone/version overlays + global heap merge + optional rerank; segments are mmap’d and compaction is background with atomic publish.

