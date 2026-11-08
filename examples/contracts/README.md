# Synthetic contract example

`synthetic.manifest.json` and `synthetic.trace.jsonl` are hand-constructed fixtures for event ordering and release gating.

One synthetic episode contains one action: completion at 80 ms, release/action start at 100 ms, termination at 150 ms. The action waits for release after inference completes.

From the repository root:

```bash
python tools/validate_contracts.py --examples
```

Expected output:

```text
VALID synthetic-run: 8 events
```

Version 1 supports **sequential episodes in a single run**. A new episode requires the previous episode's terminal boundary. Clock domains keep the same origin across episode resets: timestamps must continue monotonically instead of restarting at zero. Started requests may report late completion/drop diagnostics for an old episode, but may not release or execute old actions.

[Contract tests](../../tests/test_contracts.py) cover ordering, clocks, associations, episode boundaries, missing terminals, invalid JSON and zero-success summaries. See [schema definitions](../../schemas/README.md) for producer responsibilities.
