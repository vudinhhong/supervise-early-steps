# Provenance of the derived data

`code/artifacts/eval_suite_3471.jsonl` is a derived collection assembled from three
public sources:

| Component | Problems | Source | Licence of the source |
|---|---|---|---|
| `gsm8k-heldout` | 1200 | `openai/gsm8k`, held-out slice of the train split | MIT |
| `gsm8k-test` | 1319 | `openai/gsm8k`, test split | MIT |
| `math-heldout` | 627 | Hendrycks MATH, held-out slice of the train split | MIT |
| `math500` | 325 | `HuggingFaceH4/MATH-500` | see below |

`HuggingFaceH4/MATH-500` carries no licence declaration on its dataset card. The
subset is drawn from the test split of Hendrycks MATH by way of PRM800K, and both
of those are released under the MIT licence; we record that chain rather than
assert a licence the card does not state.

The file holds only the question text, the normalised final answer and a source
label. Its first line is a `_meta` record giving the split seed and the result of
the disjointness check against the training pool.

Model weights are not released here. The models used are from the Qwen2.5 family:
most are Apache 2.0, while Qwen2.5-3B is covered by the Qwen Research licence.
