---
name: No Timeouts; Use Efficient Latency Handling Instead
description: DocWain operations must not cut off responses on a wall-clock timer. Latency varies with query/task complexity and that's accepted; wasted or redundant work is not. Handle latency through efficiency levers (parallelism, streaming, adaptive budgets, output caps, prompt compression, caching, honest-compact fallbacks), never through timeout-and-abort.
type: feedback
originSessionId: 56b70947-9824-48b4-9a97-b3d2d50b0d88
---
When designing or fixing DocWain query paths, ingest synthesis, or any long-running operation, do NOT add hard wall-clock timeouts that abort mid-response. Latency is expected to vary with complexity. Instead, make every step efficient: parallelize retrieval and fetches, stream responses so TTFT stays low, budget context hard per intent, cap output length per intent, cache repeat retrievals, skip redundant work when signal is weak, compress prompt packs.

**Why:** Stated by the user on 2026-04-20 while reviewing the Profile-SME design latency concern: "based on the complexity the latency may vary which i understand, there should be no timeout but an efficient way to handle this latency efficiently." Aligns with the existing user memory that values accuracy over latency and enterprise-grade quality — abandoning a mid-generation response would degrade both.

**How to apply:**
- Never propose `asyncio.wait_for(...)`, `aiohttp.ClientTimeout(total=...)` for the generation call, or request-level wall-clock cutoffs that can abort a streamed response.
- Fetch/network timeouts on external I/O (URL fetches, web crawls, third-party APIs) remain fine and necessary — those are per-operation safety, not per-user-request abort.
- Frame latency work as "make each step efficient" (parallelism, budgets, caching, streaming), never "kill the slow ones."
- When latency concerns arise, reach for: parallel retrieval, intent-tuned top-K, adapter-configured context budgets, streaming-first UX, output caps per intent, Redis retrieval cache, prompt compression, honest-compact fallback when signal is weak.
- Exception: if the user explicitly asks for a timeout on a specific operation (e.g., "URL fetch should time out after 15s"), that's fine — they named the scope.
