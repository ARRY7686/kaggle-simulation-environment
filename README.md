---
title: KaggleSimEnv
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: api/server.py
pinned: false
---

# KaggleSimEnv v3

Production-grade **OpenEnv** RL environment simulating Kaggle competitions with **hierarchical action categories**, **causal dataset properties**, **failure-mode traps**, **contextual strategy scoring**, and **50+ advanced strategies**.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload
```

### Baseline agent

```bash
export OPENAI_API_KEY=sk-...
python -m baseline.run_baseline --mode local
```

---

## Architecture

```
openenvHackathon/
├── kaggle_sim_env/
│   ├── models.py         # Hierarchical categories, DatasetProperties, FailureMode
│   ├── environment.py    # Causal logic, trap detection, mitigation tracking
│   ├── tasks.py          # 5 tasks with properties, traps, context relevance
│   ├── grader.py         # 4-axis grading (perf + strategy + combo + trap)
│   ├── leaderboard.py    # Ghost competitor leaderboard
│   ├── hints.py          # Per-task hint dispensing
│   └── rewards.py        # 9-component dense reward
├── api/server.py         # FastAPI (8 endpoints)
├── baseline/run_baseline.py  # Structured phase-based agent
├── openenv.yaml / Dockerfile / requirements.txt
```

---

## Hierarchical Action Space

Actions use `category` to reduce search space:

```json
{
  "action_type": "feature_engineering",
  "parameters": {
    "category": "distribution",
    "technique": "log_transform"
  }
}
```

| Action Type | Categories → Techniques |
|---|---|
| `set_cv` | standard(kfold, repeated_kfold) · group(group_kfold, stratified_group_kfold) · temporal(time_split, combined_group_time) |
| `feature_engineering` | distribution(log_transform, normalize, quantile_features) · interaction(interaction_terms, domain_ratios) · encoding(sin_cos_encoding, target_encoding, spatial_encoding, tfidf_features) · spatial(relative_coordinates, distance_features) · signal(frequency_features, multi_layer_features, fourier_resampling) |
| `detect_shift` | detection(adversarial_validation, feature_importance_shift) · mitigation(remove_identifiers, domain_invariant_features) |
| `train_model` | tree(xgboost, lightgbm, catboost, random_forest) · linear(linear) · neural(neural_network, pretrained_backbone, temporal_cnn, transformer_encoder) |
| `handle_imbalance` | weighting(scale_pos_weight, class_weighted_loss) · calibration(calibrate_probabilities, optimize_threshold) · hierarchy(hierarchical_labels, lower_thresholds_recall) |
| `clean_data` | removal(remove_corrupted, remove_outliers, remove_leaky_features) · reconstruction(analytical_reconstruction, nan_native_model, domain_augmentation, clean_subset_training) |
| `augmentation` | geometric(geometric, rotation_invariant, image_rectification) · color(color_transform, clahe) · noise(gaussian_noise, robustness_augmentation) · domain(camera_simulation, temporal_augmentation, symmetry_augmentation, multi_view_processing) |
| `ensemble` | averaging(weighted_average, multi_seed_averaging, swa) · stacking(stacking) · diversity(diverse_features, heterogeneous) |
| `postprocess` | calibration(bias_correction, prediction_shrinkage, per_group_calibration) · domain(domain_rules, physics_constraints) · inference(tta) |
| `tune_loss` | asymmetric(asymmetric_loss, epsilon_insensitive) · uncertainty(gaussian_nll) · multi_objective(multi_task, interval_regression, quantile_regression) · weighting(sample_weighted, auxiliary_physics_loss) |
| `regularize` | weight(strong_regularization, ema, dropout) · transfer(freeze_backbone) |

Plus: `pseudo_label` (iterations), `inspect_top_solution`, `submit`

---

## Causal Dataset Properties

Each task has ground-truth properties that drive **causal** reward logic:

```python
DatasetProperties(
    has_shift=True,         # Actions addressing shift are rewarded
    has_leakage=True,       # Cleaning leaky features is critical
    has_noise_features=True, # Interaction terms on noise amplify it
    has_missing_data=True,  # Reconstruction strategies get bonus
    has_imbalance=True,     # Scale_pos_weight becomes relevant
    has_images=False,       # Image augmentation is irrelevant → penalty
    needs_physics=False,    # Physics loss is irrelevant → penalty
)
```

Actions are scored based on whether they match the dataset:

```
if dataset.has_shift and action == "adversarial_validation":
    reward += context_bonus    # Relevant!
elif not dataset.has_images and action == "geometric_augmentation":
    reward += irrelevant_penalty  # Wrong domain!
```

---

## Failure-Mode Traps

The environment contains traps that **punish common mistakes**:

| Trap | Trigger | Effect | Mitigation |
|---|---|---|---|
| kfold_on_temporal_data | Using kfold when has_shift | CV +0.08, **test -0.04** | Use time_split instead |
| ignoring_shift | Training without addressing shift | test **-0.06** | Detect shift first |
| keeping_leaky_feature | Training when has_leakage | test **-0.08** | Clean leaky features first |
| target_encoding_leakage | target_encoding on shifted data | CV +0.05, **test -0.06** | Don't use it |
| interaction_terms_on_noise | interaction_terms when noise | CV +0.05, **test -0.04** | Avoid on noisy data |
| tree_model_on_images | xgboost on image data | CV +0.04, **test -0.02** | Use pretrained_backbone |
| no_augmentation_on_images | Submit without augmentation | test **-0.04** | Apply augmentation |
| raw_heading_without_sincos | Submit without sin_cos_encoding | test **-0.03** | Encode angles properly |

Traps can be **mitigated** by taking the correct action first. The environment tracks mitigations.

---

## Grading (4 Axes)

```
final = 0.40×performance + 0.25×strategy + 0.20×combo + 0.15×trap_avoidance
```

| Component | Description |
|---|---|
| **performance** | Test score vs ghost competitors |
| **strategy** | Contextual — penalises irrelevant strategies used |
| **combo** | Fraction of synergy combos activated |
| **trap_avoidance** | 1.0 minus fraction of traps triggered |

---

## Reward Function (9 Components)

| Component | Description |
|---|---|
| `cv_improvement` | Δ CV score |
| `strategy_bonus` | +0.05 for expected strategy |
| `context_bonus` | +0.03×relevance (positive) or -0.04×relevance (negative) |
| `combo_bonus` | +0.08 per combo completed |
| `redundancy_penalty` | -0.03 × repeat count |
| `irrelevant_penalty` | -0.05 for actions with relevance ≤ -0.8 |
| `trap_penalty` | -0.08 per trap triggered |
| `overfitting_penalty` | -0.5 × gap when CV-test > 0.05 |
| `submission_bonus` | 0.5 × test_score |

---

## Tasks (5)

| Task | Difficulty | Traps | Combos | Key Challenge |
|---|---|---|---|---|
| `easy_churn` | Easy | 3 | 2 | Clean tabular, mild imbalance |
| `medium_fraud` | Medium | 3 | 3 | Shift, heavy imbalance, safety-critical |
| `hard_leaky_noisy` | Hard | 4 | 4 | Leakage, noise, missing data, shift |
| `image_quality` | Hard | 2 | 4 | Heavy-tailed, camera bias, augmentation |
| `trajectory_pred` | Hard | 2 | 4 | Multi-agent, physics, spatial-temporal |

---

## Baseline Agent

Structured multi-phase approach:
1. **Inspect** hints (1-2)
2. **Diagnose** dataset properties
3. **Clean** if needed
4. **CV** appropriate for domain
5. **Features** domain-relevant only
6. **Train** right model family
7. **Tune** imbalance/loss
8. **Ensemble** (1-2 techniques)
9. **Submit**

Keeps actions to 8-15 total. Uses hints to inform decisions.

---

## Docker

```bash
docker build -t kaggle-sim-env .
docker run -p 7860:7860 kaggle-sim-env
```

---

## License

MIT
