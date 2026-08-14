# RL Distributed Dev Reward Design

## Problem

At optimizer step 100, all four ranks completed distributed dev CE. Rank 0 then
ran all 128 greedy reward samples alone while ranks 1-3 entered the checkpoint
RNG `gather_object`. Qwen semantic judging issues one synchronous request per
concrete attribute pair, so rank 0 remained outside the collective for more
than the 600-second process-group timeout. The NCCL watchdog aborted all ranks
before `checkpoint-100` could be written.

## Design

The fixed first 128 dev rows retain their original indices and are assigned
round-robin across all ranks. Each rank generates captions and extracts
attributes using its local Florence replica, then computes reward records for
its shard. Rank-local success or error payloads are gathered; rank 0 validates
that every original index occurs exactly once and restores the original order.
Rank 0 remains the only writer of predictions, metrics, and checkpoints.

The Qwen judge client exposes a bounded prefetch operation. Before classifying
a batch, the trainer enumerates and deduplicates all fixed-field and `extra`
value comparisons, then executes them in a thread pool. Normal similarity
lookups subsequently hit the existing cache. Formal concurrency is 16 requests
per rank, so four ranks submit at most 64 concurrent requests to the judge
service configured for 128 sequences. Lexical matching remains synchronous and
network-free.

The DDP process-group timeout becomes an explicit 60-minute invariant. This is
defense in depth for uneven shard latency and checkpoint serialization; it is
not the primary fix. Any rank-local evaluation error is gathered and broadcast
so all ranks fail coherently instead of leaving peers in a collective.

## Verification And Restart

Unit tests cover round-robin partitioning, exact ordered merge, duplicate and
missing index rejection, error propagation, comparison enumeration, deduped
concurrent prefetch, and timeout configuration. After the full suite passes, a
four-GPU one-step run performs real distributed dev reward against both Qwen
services. The failed run and logs are archived, then the Qwen and lexical formal
runs restart serially from the V4B final checkpoint on physical GPUs 3-6.

