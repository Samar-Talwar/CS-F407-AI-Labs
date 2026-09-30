# Prompt Log

Date | Week | Prompt | Changes made after review
--- | --- | --- | ---
2026-09-30 | 1 | "Generate minimal PyTorch code for a 2-2-1 neural network trained on the four XOR examples (0,0→0, 0,1→1, 1,0→1, 1,1→0) with sigmoid hidden units, sigmoid output via BCEWithLogitsLoss, random weight initialisation, full-batch training for 5000 steps. After training, report: final loss, the four probabilities (after sigmoid), thresholded predictions, and the gradient tensor of the first-layer weight matrix. Set random seed 42 for reproducibility. Explain each test in one sentence." | None needed — LLM output already satisfied all constraints. Wrapped into reusable functions and added CLI/test harness.