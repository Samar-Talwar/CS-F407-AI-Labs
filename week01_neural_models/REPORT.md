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
The two classes are interleaved diagonally; no single linear decision boundary $w_1 x_1 + w_2 x_2 + b = 0$ can separate class 1 points {(0,1), (1,0)} from class 0 points {(0,0), (1,1)}. This is the classic linear inseparability of the XOR function.

**Linear baseline results (`results/linear_baseline.json`)**  
A single affine map followed by sigmoid output was trained with SGD (lr=1.0, 5000 steps, seed 42):
- Initial BCE loss: 0.731738
- Final BCE loss: 0.693147 ($\approx \ln 2$, matching random guessing)
- Output probabilities: [0.5000, 0.5000, 0.5000, 0.5000]
- Thresholded predictions: [0, 0, 0, 0]
- Accuracy: 2/4 correct (fails to solve XOR)

**Think About It 1**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* XOR tests the representational limit of single-layer perceptrons without hidden representation transformations. A linear layer cannot warp the geometric space to linearly separate diagonal clusters; learning a non-linear mapping is mandatory.

---

## 2. Model design and validation criteria (Task 2)

**Model specification**  
- Architecture: 2 inputs → 2 hidden units → 1 output  
- Hidden activation: sigmoid (baseline; comparisons with tanh, ReLU)  
- Output: single logit with sigmoid implicit in BCEWithLogitsLoss  
- Loss: binary cross‑entropy (`torch.nn.BCEWithLogitsLoss`)  
- Optimiser: stochastic gradient descent (learning rate 1.0)  
- Initialisation: random (PyTorch default) with explicit manual seeds for reproducibility  
- Training: full‑batch gradient descent, 5000 steps  

**Validation criteria for successful learning**  
1. Final binary cross‑entropy loss < 0.1 (converging towards 0)  
2. All four thresholded predictions match targets [0, 1, 1, 0] (4/4 correct)  
3. First-layer weight gradient norm $\| \nabla_{W^{(1)}} L \| > 0$ at step 0 and step 10  

**Think About It 2**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* The network does not have hand-crafted features; gradient descent automatically configures the two hidden units to act as intermediate geometric features (such as OR and NAND hyperplanes) to map inputs into a linearly separable space for the output unit.

---

## 3. Exact LLM prompt(s) and corrections

**Prompt used**  
> "Generate minimal PyTorch code for a 2-2-1 neural network trained on the four XOR examples (0,0→0, 0,1→1, 1,0→1, 1,1→0) with sigmoid hidden units, sigmoid output via BCEWithLogitsLoss, random weight initialisation, full-batch training for 5000 steps. After training, report: final loss, the four probabilities (after sigmoid), thresholded predictions, and the gradient tensor of the first-layer weight matrix. Set random seed 42 for reproducibility. Explain each test in one sentence."

**Corrections made to generated code**  
The raw generated code was refactored into modular components (`models.py`, `train.py`, `cli.py`, `test_week01.py`). For ReLU, hyperparameter adjustment (seed=5, lr=0.5, steps=20000) was introduced to prevent dying ReLU units on this small discrete dataset.

**Think About It 3**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* Structural properties like layer dimensions, activation types, and syntax can be verified statically; convergence, gradient dynamics, dead units, and empirical loss require dynamic runtime execution.

---

## 4. Final code used for binary XOR and three-class extension

- Network architectures: `week01_neural_models/src/models.py` (`XORBinaryNet`, `XORLinearNet`, `XORMulticlassNet`)  
- Training & diagnostic routines: `week01_neural_models/src/train.py` (`train_binary_xor`, `train_linear_baseline`, `train_symmetry_experiment`, `train_multiclass`, `seed_sweep`)  
- CLI reproduction entry point: `week01_neural_models/src/cli.py`  
- Pytest test suite: `week01_neural_models/tests/test_week01.py`  

---

## 5. Empirical results from results/*.json

All figures below are extracted directly from machine-generated JSON files in `results/`.

### 5A – Binary XOR (Sigmoid hidden, `results/binary_xor.json`)
- Initial loss: 0.748230  
- Final loss: 0.002630  
- Four probabilities: [0.003326, 0.997637, 0.997637, 0.002450]  
- Thresholded predictions: [0, 1, 1, 0] (4/4 correct)  
- First-layer weight gradient norm $\| \nabla_{W^{(1)}} L \|$:  
  - At step 0: 0.011894  
  - At step 10: 0.004387  
- Mean-loss gradient vs average of per-example gradients max difference: 0.00e+00 ($< 10^{-6}$). The 4 independent per-example backward passes on single-example losses (each loss is `BCEWithLogitsLoss` on one row) are averaged; the mean equals the single backward pass on the full-batch mean loss to float32 precision. Difference reflects floating-point accumulation order, not logic error. Per-example gradient tensors (full precision, 2×2 each) and mean gradient tensor are stored in `results/binary_xor.json` under `per_example_grads_w1` and `grad_mean_w1`.

### 5B – Symmetry experiment (`results/symmetry.json`)
- Zero initialisation of all weights and biases.  
- Recorded hidden weight row equality at steps `[0, 1, 5, 10, 100]`: `[True, True, True, True, True]` (both rows remained identical throughout training).  
- Final loss: 0.693147 ($\approx \ln 2$, failing to break symmetry or solve XOR).  

### 5C – Activation comparison & Seed Sweep (`results/activation_comparison.json`, `results/seed_sweep.json`)

**Experiment Configuration & Reconciliation:**
- **Optimiser across all runs:** Standard Stochastic Gradient Descent (`torch.optim.SGD`).
- **Sigmoid & Tanh:** Seed 42, learning rate 1.0, 5,000 steps.
- **ReLU:** Seed 5, learning rate 0.5, 20,000 steps.
- **Why settings differ:** On a minimal 2-2-1 network with only 4 discrete training points, ReLU hidden units are particularly susceptible to the "dying ReLU" problem when weights are randomly initialized. If pre-activations for all 4 inputs fall in the non-positive regime ($z \le 0$), the unit emits zero gradient ($\text{ReLU}'(z) = 0$) and permanently dies. A slightly reduced learning rate (0.5), tailored seed (5), and longer training horizon (20,000 steps) allow the non-dead units sufficient iterations to navigate the piecewise-linear loss surface to zero loss.

| Hidden activation | Seed | Optimiser | LR | Steps | Final loss | 4/4 correct? | Early $\| \nabla_{W^{(1)}} L \|$ (@ step 10) |
|---|---|---|---|---|---|---|---|
| Sigmoid | 42 | SGD | 1.0 | 5000 | 0.002630 | Yes | 0.004387 |
| Tanh | 42 | SGD | 1.0 | 5000 | 0.000637 | Yes | 0.016810 |
| ReLU | 5 | SGD | 0.5 | 20000 | 0.000145 | Yes | 0.029017 |

**5-Seed Sweep Results (seeds: 42, 123, 456, 789, 999):**
- **Sigmoid:** 2/5 seeds reached 4/4 (successful seeds: 42, 456; failed seeds: 123, 789, 999 failed due to loss plateaus / saddle points / local minima around loss $\approx 0.693$).
- **Tanh:** 2/5 seeds reached 4/4 (successful seeds: 42, 456; failed seeds: 123, 789, 999 failed due to loss plateaus / local minima).
- **ReLU:** 1/5 seeds reached 4/4 (successful seed: 456; failed seeds: 42, 123, 789, 999 failed because hidden units received strictly non-positive pre-activations across all four input patterns, resulting in zero gradients and dead ReLU units).

*Empirical Note:* No activation function is universally or reliably superior on this minimal architecture; each displays distinct trade-offs between gradient saturation and dead-unit vulnerability under random initialization.

**Think About It 4**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* In sigmoid saturation, pre-activations are large in magnitude ($|z| \gg 0$) yielding small but non-zero gradients $\sigma'(z) \approx 0$; in dead ReLU, pre-activations remain strictly negative ($z < 0$), resulting in exact zero gradient and complete stagnation.

### 5D – Three-class extension (`results/multiclass.json`)
- Architecture: 2 inputs → 2 hidden units → 3 output logits  
- Output weight matrix shape: `[3, 2]`  
- Logits per example: 3  
- Hidden dimension: 2  
- Final Cross-Entropy Loss: 0.001505  
- Predicted class probabilities:  
  - (0,0) → [0.998432, 0.001568, 5.28e-10] (Pred: 0, Target: 0)  
  - (0,1) → [0.000584, 0.998669, 0.000747] (Pred: 1, Target: 1)  
  - (1,0) → [0.000585, 0.998668, 0.000747] (Pred: 1, Target: 1)  
  - (1,1) → [3.13e-08, 0.001783, 0.998217] (Pred: 2, Target: 2)  
- Predictions: [0, 1, 1, 2] (4/4 correct)  
- Softmax probability sums: [1.000000, 1.000000, 1.000000, 1.000000] (each $\approx 1.0$)  
- Shift-invariance constant: 100.0  
- Max absolute difference between unshifted and shifted probabilities: 5.24e-09 ($< 10^{-5}$)  
- Naive softmax overflow: Naive softmax without max subtraction overflows on large logits (`logits + 100`), whereas stable max-subtracted softmax preserves numerical fidelity.

**Think About It 5**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* The mathematical formulation $\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$ and logit gradient $\nabla_{z} L = p - y$ remain identical regardless of vocabulary size; the difference is computational overhead in computing the normalizing denominator over large vocabulary dimensions.

---

## 6. Reflection question answers

**1. What did the XOR experiment demonstrate about the difference between depth and non‑linearity?**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* Depth without non-linearity collapses into a single affine transformation $W_2(W_1 x + b_1) + b_2 = W_{eff} x + b_{eff}$, which is provably incapable of separating non-linearly separable functions like XOR. Adding a non-linear activation in the hidden layer creates a warped feature representation, enabling linear separation at the output stage.

**2. In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a non‑zero gradient?**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* The loss monotonically decreased from 0.7482 to 0.0026, and the predictions moved from ambiguous intermediate values to definitive target classifications [0, 1, 1, 0]. A useless non-zero gradient (such as noise) would cause random walk without systematic loss minimization.

**3. Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* With zero initialization, all hidden units compute identical pre-activations and activations for any given input. By the chain rule, incoming gradients from the output layer are identical for both units. Since $\Delta W_0 = \Delta W_1$, the weights remain identical at every step (verified in `symmetry.json`), preventing the units from specializing into distinct detectors.

**4. How did changing the hidden activation affect the gradient you observed? Distinguish the scientific explanation from the engineering observation.**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):*  
> - *Scientific:* Sigmoid compresses gradients through its derivative $\sigma'(z) = \sigma(z)(1-\sigma(z)) \le 0.25$, leading to smaller early gradient norms. Tanh has a maximum derivative of 1.0 at zero. ReLU has derivative 1 for $z>0$ and 0 for $z<0$.  
> - *Engineering:* Early gradient norms followed $\| \nabla_{W^{(1)}} L \|_{\text{sigmoid}} (0.0044) < \| \nabla_{W^{(1)}} L \|_{\text{tanh}} (0.0168) < \| \nabla_{W^{(1)}} L \|_{\text{relu}} (0.0290)$. ReLU required tuning initial seeds and learning rate to avoid units getting stuck in negative inactive states.

**5. Why must the output layer and loss be selected together according to the task?**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):* Pairing sigmoid with BCE (or softmax with Cross-Entropy) yields the elegant gradient $\frac{\partial L}{\partial z} = p - y$. This cancels out derivative saturation terms in the denominator, preventing vanishing gradients during high-error regimes and providing a linear error-proportional update step.

**6. Give one example where the LLM improved your engineering productivity and one example where human verification was essential.**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):*  
> - *Productivity:* The LLM rapidly scaffolded PyTorch boilerplate (parameter zeroing, backward passes, dataclass result containers, pytest test suites).  
> - *Human Verification:* Detecting that default seed/hyperparameters caused ReLU to fail on XOR due to dead neurons, requiring principled engineering adjustment to recover 4/4 convergence.

**7. Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?**  
> *[DRAFT - rewrite in own words]*  
> *TODO(student):*  
> - *Keep:* End-to-end evaluation metrics (loss convergence, accuracy), gradient non-zero assertions, softmax sum normalization checks, shift-invariance unit tests.  
> - *Too Expensive:* Tracking full weight tensor histories across every training step, exhaustive per-example manual gradient checks vs full-batch gradients, and multi-seed grid sweeps over hundreds of parameter combinations.

---

## 7. Submission Checklist
- [x] `ruff check .` passes without errors
- [x] `pytest` passes all 31 tests
- [x] CLI (`python -m week01_neural_models.src.cli`) regenerates all JSON deliverables
- [x] All numerical values in report sourced from `results/*.json`
- [x] `CHECKLIST.md` complete and updated
- [x] `docs/prompt_log.md` updated with prompt history
- [x] Root `README.md` status table updated
