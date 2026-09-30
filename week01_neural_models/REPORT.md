# CS F407 Lab, Week 1 | Author: Samar Talwar | Not licensed for reuse or submission by others.

## 1. Task 1 problem specification and linear-separability explanation

**Problem specification**
- Input space X = {(0,0), (0,1), (1,0), (1,1)} ⊂ ℝ²  
- Output space Y = {0, 1}  
- Training set:  
  (0,0) → 0  
  (0,1) → 1  
  (1,0) → 1  
  (1,1) → 0  

**Why one straight line cannot separate the classes**  
Plotting the four points in the (x1, x2)-plane:  
```
x2
1 |  ● (0,1)   ● (1,1)
  |
0 |  ● (0,0)   ● (1,0)
  +----------------→ x1
    0            1
```
The two classes are interleaved; no single line y = w₁x₁ + w₂x₂ + b can put (0,1) and (1,0) on one side while (0,0) and (1,1) are on the other. This is the linear inseparability of XOR.

**Prediction for affine + sigmoid**  
A single affine map followed by sigmoid can only produce decision boundaries that are straight lines in input space. Therefore the network will converge to random guessing (loss ≈ ln 2 ≈ 0.693) and predict ~0.5 for all four inputs.

## 2. Model design and validation criteria (Task 2)

**Model specification**  
- Architecture: 2 inputs → 2 hidden units → 1 output  
- Hidden activation: sigmoid (baseline; later tanh, ReLU)  
- Output: single logit with sigmoid implicit in BCEWithLogitsLoss  
- Loss: binary cross‑entropy (via `torch.nn.BCEWithLogitsLoss`)  
- Optimiser: stochastic gradient descent (learning rate 1.0)  
- Initialisation: random (PyTorch default) with manual seed for reproducibility  
- Training: full‑batch, 5000 steps  

**Validation criteria for successful learning**  
1. Final binary cross‑entropy loss < 0.1 (close to 0)  
2. All four thresholded predictions match targets [0,1,1,0]  
3. At least one gradient component (e.g. ∂L/∂W₁₁₁) is significantly non‑zero after backward()  

## 3. Exact LLM prompt(s) and corrections

**Prompt used**  
> "Generate minimal PyTorch code for a 2-2-1 neural network trained on the four XOR examples (0,0→0, 0,1→1, 1,0→1, 1,1→0) with sigmoid hidden units, sigmoid output via BCEWithLogitsLoss, random weight initialisation, full-batch training for 5000 steps. After training, report: final loss, the four probabilities (after sigmoid), thresholded predictions, and the gradient tensor of the first-layer weight matrix. Set random seed 42 for reproducibility. Explain each test in one sentence."

**Corrections made to generated code**  
None. The first generated snippet already satisfied all constraints; I only wrapped it into a reusable `train_binary_xor()` function and added the CLI/test harness.

## 4. Final code used for binary XOR and three-class extension

See `src/models.py` (network definitions), `src/train.py` (training loops), and `src/cli.py` (entry point that regenerates every result).

## 5. Requested loss, prediction, gradient, symmetry, and activation results

All numbers below come from the JSON files in `results/`, produced by `python -m week01_neural_models.src.cli`.

**5A – Binary XOR (sigmoid hidden)**  
- Initial loss: 0.748230  
- Final loss: 0.002630  
- Four probabilities: [0.0033, 0.9976, 0.9976, 0.0024]  
- Thresholded predictions: [0, 1, 1, 0] ✓  
- ‖∇W(1)‖ at step 10: 0.004387  
- First-layer weight gradient matrix (example):  
  [[ 0.003267,  0.002862],  
   [-0.000588,  0.000191]]  

**5B – Symmetry experiment (zero initial weights)**  
- Hidden weight rows remained identical throughout training (see `results/symmetry.json`)  
- Final loss: 0.693147 (≈ ln 2, random guessing)  

**5C – Activation comparison**  

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W(1)‖ | Notes |
|-------------------|------------|--------------|----------------|-------|
| Sigmoid           | 0.002630   | Yes          | 0.004387       | Baseline |
| Tanh              | 0.000637   | Yes          | 0.016810       | Faster, larger gradients |
| ReLU              | 0.000145   | Yes          | 0.029017       | Required seed=5, lr=0.5, steps=20000 to avoid dead‑unit problem |

**5D – Three-class extension**  
- Final loss: 0.001505  
- Predicted class probabilities:  
  (0,0) → [0.9984, 0.0016, 5.3e-10]  
  (0,1) → [0.0006, 0.9987, 0.0007]  
  (1,0) → [0.0006, 0.9987, 0.0007]  
  (1,1) → [3.1e-8, 0.0018, 0.9982]  
- Predicted class labels: [0, 1, 1, 2] ✓  
- Softmax probability sums: [1.000000, 1.000000, 0.999999, 1.000000] (each ≈ 1)  
- Shift-invariance verified: adding 100 to all logits leaves probabilities unchanged (within 1e-5).

## 6. Reflection question answers

**1. What did the XOR experiment demonstrate about the difference between depth and non‑linearity?**  
Depth without non‑linearity (affine → affine → … → affine) collapses to a single affine map, which cannot represent XOR. Adding a non‑linear hidden activation breaks linearity and allows the network to learn the XOR function with only one hidden layer. Thus, non‑linearity—not depth—is the key ingredient.

**2. In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a non‑zero gradient?**  
The loss decreased monotonically from 0.748 to 0.0026, and the network achieved perfect classification ([0,1,1,0]). A useless gradient would leave loss near ln 2 and predictions near 0.5. The observed gradient direction consistently reduced loss, proving it pointed toward useful parameter updates.

**3. Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?**  
With identical weights and biases, both hidden units compute exactly the same activation for any input. During backprop, they receive identical gradients (∂L/∂W₁₍ᵢ₎ identical for i=1,2), so their weights remain locked together. Symmetry breaking requires asymmetric initialisation so that hidden units specialise on different features.

**4. How did changing the hidden activation affect the gradient you observed? Distinguish the scientific explanation from the engineering observation.**  
*Scientific*: Sigmoid and tanh have non‑zero derivatives in their operating regimes, propagating meaningful error signals. ReLU’s derivative is either 0 (dead unit) or 1 (active), which can yield larger gradients when active but risks vanishing if pre‑activations go negative.  
*Engineering*: For the same seed/lr/steps, ‖∇W(1)‖ was smallest with sigmoid (~0.004), larger with tanh (~0.017), and largest with ReLU (~0.029) *once we tuned hyperparameters to avoid dead units*. The larger ReLU gradient reflects its unit slope in the active regime.

**5. Why must the output layer and loss be selected together according to the task?**  
The pairing (sigmoid, binary cross‑entropy) or (softmax, cross‑entropy) makes the logit gradient exactly (p − y), which is simple, numerically stable, and ensures gradient magnitude matches prediction error. Mismatched pairs (e.g. sigmoid output with MSE loss) produce vanishing gradients or incorrect gradients, hindering learning.

**6. Give one example where the LLM improved your engineering productivity and one example where human verification was essential.**  
*LLM productivity*: The LLM generated a correct, constraint‑respecting 2-2-1 training loop in seconds, sparing me boilerplate wiring (optimiser loop, zero_grad, step).  
*Human verification essential*: The LLM’s default ReLU run failed to classify all four points correctly (dead‑unit problem). Only by inspecting predictions and adjusting seed/lr/steps (permitted engineering changes) did we recover a working ReLU experiment.

**7. Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?**  
*Keep*: Final loss value, prediction correctness, gradient non‑zero checks, softmax‑sum-to-one, shift‑invariance. These are O(1) or O(batch size) and scale trivially.  
*Too expensive*: Exhaustive finite‑difference gradient checks (O(parameters²)), symmetry experiments that require tracking all weights over time, and ablation studies that retrain dozens of hyperparameter settings. These grow quadratically or worse with model size.

---
**Submission checklist**  
- [x] `ruff check .` passes  
- [x] `pytest` passes  
- [x] CLI regenerates all results from a fresh process  
- [x] Numbers in this report come from `results/`  
- [x] CHECKLIST.md updated and ticked  
- [x] docs/prompt_log.md updated  
- [x] Root README status table updated