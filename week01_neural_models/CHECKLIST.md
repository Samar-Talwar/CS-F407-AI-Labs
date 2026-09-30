# Week 1 – Neural Models: Checklist

Every requirement from `docs/lab_sheets/neur_models_lab_ex.pdf`, mapped to deliverables.

## Task 1 – Understand the problem before coding
| # | Requirement | File / Test / Result |
|---|---|---|
| 1.1 | Input space X, output space Y, 4 labelled examples | REPORT.md §1 |
| 1.2 | Sketch of 4 points in x1-x2 plane labelled by class | REPORT.md §1 (ASCII) |
| 1.3 | Why one straight boundary cannot separate classes | REPORT.md §1 |
| 1.4 | Prediction: single affine + sigmoid outcome | REPORT.md §1 |
| TAI-1 | Think About It: what claim about representation does XOR test? | REPORT.md §Think-About-It-1 |

## Task 2 – Design the intelligent agent
| # | Requirement | File / Test / Result |
|---|---|---|
| 2.1 | Model spec: 2-2-1, hidden activation, sigmoid out, BCE, GD | REPORT.md §2, src/models.py |
| 2.2 | Why hidden nonlinearity is scientifically necessary | REPORT.md §2 |
| 2.3 | Why sigmoid + BCE is sensible pairing | REPORT.md §2 |
| 2.4 | Three checks for successful learning defined | REPORT.md §2 |
| TAI-2 | Think About It: what determined what hidden units compute? | REPORT.md §Think-About-It-2 |

## Task 3 – LLM-generated first implementation
| # | Requirement | File / Test / Result |
|---|---|---|
| 3.1 | Exact LLM prompt recorded | docs/prompt_log.md, REPORT.md §3 |
| 3.2 | Code with all constraints (4 examples, 2-2-1, BCEWithLogitsLoss, random init, full-batch, print loss/preds/grads) | src/models.py, src/train.py |
| 3.3 | Identify forward pass, loss, backward(), optimiser step in code | REPORT.md §3 |
| 3.4 | Two changes made (if any) | REPORT.md §3 |
| TAI-3 | Think About It: what can you verify without running vs requires execution? | REPORT.md §Think-About-It-3 |

## Task 4 – Execute, test, diagnose
### Part A – Basic learning check
| # | Requirement | File / Test / Result |
|---|---|---|
| 4A.1 | Initial and final loss recorded | results/binary_xor.json |
| 4A.2 | Four final probabilities and thresholded predictions | results/binary_xor.json |
| 4A.3 | All 4 labels correct | tests/test_binary_xor.py |

### Part B – Backpropagation check
| # | Requirement | File / Test / Result |
|---|---|---|
| 4B.1 | Gradients of first-layer weights inspected | results/binary_xor.json |
| 4B.2 | Explain parameter.grad = ∂L/∂W(1) | REPORT.md §4B |
| 4B.3 | Why gradient = average of example-wise gradients (mean loss) | REPORT.md §4B |

### Part C – Symmetry experiment
| # | Requirement | File / Test / Result |
|---|---|---|
| 4C.1 | All weights = 0, train, inspect hidden weight rows | results/symmetry.json |
| 4C.2 | Record whether rows remain identical | tests/test_symmetry.py |
| 4C.3 | Explain using identical output ⇒ identical gradient | REPORT.md §4C |

### Part D – Activation experiment
| # | Requirement | File / Test / Result |
|---|---|---|
| 4D.1 | Train with sigmoid, tanh, ReLU; record final loss, 4/4 correct, early ‖∇W(1)L‖ | results/activation_comparison.json |
| 4D.2 | Result table | REPORT.md §4D |
| 4D.3 | Explain observed differences (no universal "best" claim) | REPORT.md §4D |
| TAI-4 | Think About It: distinguishing sigmoid saturation vs ReLU dead unit | REPORT.md §Think-About-It-4 |

## Task 5 – Three-class extension
| # | Requirement | File / Test / Result |
|---|---|---|
| 5.1 | 3-class task: class 0=(0,0), class 1=(0,1)/(1,0), class 2=(1,1) | src/models.py, src/train.py |
| 5.2 | Predict: shape of final weight matrix | REPORT.md §5 |
| 5.3 | Predict: number of logits per example | REPORT.md §5 |
| 5.4 | Why softmax probabilities sum to 1 | REPORT.md §5 |
| 5.5 | Why logit gradient = p − y | REPORT.md §5 |
| 5.6 | Predicted class probs for all 4 inputs | results/multiclass.json |
| 5.7 | Softmax vector sums to ≈1 verified | tests/test_multiclass.py |
| 5.8 | Optional: add constant to logits, probs unchanged | results/multiclass.json, tests/test_multiclass.py |
| 5.9 | Explain stable softmax (subtract max logit) | REPORT.md §5 |
| TAI-5 | Think About It: what stays same scaling to large vocab? | REPORT.md §Think-About-It-5 |

## Reflection Questions (6 total + bonus 7th)
| # | Question | Location |
|---|---|---|
| R1 | Depth vs nonlinearity | REPORT.md §Reflections |
| R2 | Evidence backprop supplied useful signal | REPORT.md §Reflections |
| R3 | Why zero init prevents distinct features | REPORT.md §Reflections |
| R4 | Activation effect on gradient | REPORT.md §Reflections |
| R5 | Output layer + loss selected together | REPORT.md §Reflections |
| R6 | LLM helped productivity + human verification essential | REPORT.md §Reflections |
| R7 | Which tests scale up, which become expensive | REPORT.md §Reflections |

## Submission Items
| # | Item | Location |
|---|---|---|
| S1 | Task 1 problem spec + linear-separability explanation | REPORT.md §1 |
| S2 | Model design + validation criteria (Task 2) | REPORT.md §2 |
| S3 | Exact LLM prompts + corrections note | REPORT.md §3, docs/prompt_log.md |
| S4 | Final code: binary XOR + 3-class extension | src/ |
| S5 | Loss, prediction, gradient, symmetry, activation results | results/ |
| S6 | Reflection question answers | REPORT.md §Reflections |

## Meta
- [ ] `ruff check .` passes
- [ ] `pytest` passes
- [ ] CLI reproduces all results from fresh run
- [ ] README.md has run instructions
- [ ] Root README status table updated
- [ ] docs/prompt_log.md updated
