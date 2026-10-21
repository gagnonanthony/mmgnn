# Library guidelines

Patterns for libraries this repo actually uses. This is not a general style guide.

## PyTorch `torch.load`

Checkpoints and saved datasets are written with `torch.save` and contain more than tensors:

- Training checkpoints store `model_state_dict`, `optimizer_state_dict`, `history`, and `model_config`.
- `load_dataset` stores a list of PyTorch Geometric `Data` objects plus metadata.

`weights_only=True` cannot load those objects. Every `torch.load` in this package passes `weights_only=False` on purpose, including under PyTorch 2.6+ where `True` is the default.

Load only files this project wrote. Do not point these loaders at a checkpoint or dataset you did not produce. Do not switch a call to `weights_only=True` unless the saved payload is tensors only.
