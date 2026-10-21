## Step 4 — Devil’s Advocate

### What is the laziest thing in this diff?
Shipping five CLI entry points and proving them only with `--help`. That is ceremony, not coverage. Second place: `cli/evaluate.py` (~534 lines) as a single `main()` script dumped into the package — untestable orchestration pretending to be a library module.

### What will break first in production?
1. **`checkpoint_path.parent.parent`** (evaluate / inference / ablate). Assumes `…/run/checkpoints/file.pt`. Point a checkpoint anywhere else and you silently pick the wrong run root, wrong normalization stats, wrong outputs.
2. **`primary_dmri_key` must live in `edge_attributes`**. Easy to omit in JSON; failure is late, during graph build, not at config validate.
3. **`torch.load(..., weights_only=False)` everywhere**. Fine for your own artifacts; one shared/untrusted `.pt` and you own a pickle gadget. No integrity check, no version stamp on the payload beyond whatever you stuffed in metadata.
4. **Train vs evaluate config drift**. Evaluate rebuilds the dataset from a separate JSON and filters evaluation keys out of the model config by set difference. Wrong edge features / normalize flags → plausible metrics on the wrong graphs. Help smokes will not catch this.

### Convenience hacks that don’t belong
- **Deep mocks in explain tests** (`build_encoder` / `build_model` / `BrainGraphDataset` patched). You are unit-testing glue around stubs, then claiming the load path works.
- **Fake models for trainer/ablation** are fine at boundaries; using them to skip real `state_dict` round-trips for `load_frozen_gae` is not.
- **God-script CLIs** instead of thin wrappers over library functions — forces every regression to be a full CLI integration or nothing.

### What will confuse someone in 6 months?
- **`mmgnn.py` / `backbone.py` deleted** with GAE/VGAE as the whole story — was multimodal fusion abandoned, deferred, or renamed? Diff doesn’t say.
- **Ownership of helpers**: `select_graphs` lives under `explain.utils` but `inference` imports it; embeddings live under `inference` and viz/explain import those. Cross-layer imports with no shared `mmgnn.data`/`mmgnn.io` home.
- **Cache naming via 4-byte blake2** plus a long payload — short digest, opaque folder names; hard to debug “why didn’t my config change reload?”
- **Review standards file is the review prompt**; `library-guidelines.md` never existed — so “we followed best practices” is circular.

### Credit where due (so the roast is fair)
EarlyStopping baseline bug, softplus docstring, silent split drops, `MODEL_KEYS`, broken `prepare_brain_graph_dataset`, and duplicated embedding loaders look cleaned up. The remaining risk is operational: layout assumptions, config/cache coupling, and tests that don’t exercise the CLIs you just invented.

**Stop.** Confirm to proceed to Step 5 (Verdict punch list).