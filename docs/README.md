# Model Documentation

This directory contains detailed documentation for each supported transformer model architecture,
explaining how the Transformer Roofline Analyzer computes hardware requirements.

## Supported Models

| Model | Documentation | Description |
|-------|---------------|-------------|
| LLaMA-2/3 | [llama.md](llama.md) | Standard dense transformer with GQA |
| LLaMA-4 | [llama4.md](llama4.md) | Mixture of Experts (MoE) architecture |

## Document Structure

Each model documentation includes:

1. **Overview** - Model type, architecture, parser, and config format
2. **Architecture** - Transformer block structure and key features
3. **Layer Analysis** - Breakdown of each layer and its operation type
4. **Config Fields** - Required fields from HuggingFace config.json
5. **Hardware Metrics Formulas** - Detailed formulas for FLOPs and bandwidth
6. **Storage Requirements** - Weights, KV-cache, and additional storage

## Adding Documentation for New Models

When adding support for a new model architecture:

1. Create `docs/<model_type>.md` following the structure above
2. Document all layers and their hardware metrics formulas
3. List all config fields used by the parser
4. Include example CLI usage
5. Update this README to add the new model to the table

## Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development setup and contribution guidelines
- [core/README.md](../core/README.md) - Base parser framework and utilities
- [parsers/README.md](../parsers/README.md) - Parser implementation guide
