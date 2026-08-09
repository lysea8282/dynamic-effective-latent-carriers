# Checkpoints

Six checkpoints are packaged: three development seeds and three independent replication seeds. Each public checkpoint preserves the exact tensor values and normalization constants while replacing workflow-specific descriptive metadata.

`checkpoints/CHECKPOINT_MANIFEST.json` records the public file hash, tensor-bundle hash, seed, parameter count, carrier dimension, and consumers. The model topology has 496,200 parameters and a 192-dimensional latent carrier.

Checkpoint archive bytes differ from their source containers because public metadata was rewritten. Tensor-bundle hashes are the scientific identity check.
