---
license: apache-2.0
base_model: Qwen/Qwen3-4B
library_name: transformers
pipeline_tag: text-generation
tags:
  - citation-recovery
  - reinforcement-learning
  - grpo
---

# AttriCite-4B

AttriCite-4B is a full-parameter GRPO fine-tune of Qwen3-4B for open-ended
citation recovery with the CiteGuard retrieval environment. Given a passage
containing `[CITATION]`, the model searches and inspects Semantic Scholar and
selects a paper returned by the retrieval tools.

Repository, paper, dataset, and model artifacts are provided through the
anonymous supplementary-material links associated with the submission.
Permanent URLs and revisions are intentionally withheld during double-blind
review and will be added to the camera-ready release.

## Training

- Base model: `Qwen/Qwen3-4B`
- Objective: full-parameter GRPO with binary normalized-title-match reward
- Training/validation examples: 350/60, from 2024 source papers
- Rollouts per prompt: 8
- Prompt batch size: 2
- Learning rate: 1e-6
- KL coefficient: 1e-3
- Sampling: temperature 0.7, top-p 0.95
- Maximum actions/context: 5 / 32,768 tokens
- Selected checkpoint: step 475, selected only on the 60-item validation set
- Hardware: 4 NVIDIA L40S GPUs

The exact launcher, agent loop, tool schema, reward, and data conversion are in
`training/verl/`. Generic Qwen thinking mode is disabled in training and
evaluation.

## Intended use and limitations

This is a research model for retrieving candidate scholarly sources. Its output
is not evidence that a citation supports a claim. Users must inspect the source,
verify entailment and bibliographic identity, and follow the original work's
license. Results depend on the evolving Semantic Scholar index and network
availability. The training data is concentrated in five computer-science venues;
biomedical results are preliminary. Published citations are treated as a proxy
for author intent and can themselves be incomplete or incorrect.

## License

The model weights are released under Apache-2.0. Repository code is MIT. The
dataset license applies only to metadata and annotations created by the authors;
third-party publication text retains its original copyright.
