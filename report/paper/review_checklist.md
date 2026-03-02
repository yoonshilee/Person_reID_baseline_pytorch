# Reid Experiment Report — Review Checklist

Based on `requirement.md` requirements and `milestones.md` experimental data.

---

## Category 1: Mandatory Gaps (Requirements Hard Requirements)

- [x] **Q3 (output dimension) added to Quick Questions** — "Why is output dimension `batchsize × 751`?" is missing. milestones.md M8.1 has the answer: the classifier outputs logits over the number of training identities (751 for Market, 702 for Duke), reflecting the closed-set classification structure. *(Fixed)*
- [x] **`train_all` impact discussion expanded** — requirement.md explicitly says "discuss its impact". The original two sentences are insufficient; needs to cover: no validation split → cannot use early stopping, full training set → potentially better generalization but harder to monitor overfitting, and link to the convergence trajectory (val_acc from 0.52 to 0.98). *(Fixed)*
- [x] **Loss variant analysis paragraph missing** — Table 2 has all data but no text analyzes Circle/Instance/Triplet vs. baseline. Circle loss mAP (0.6128) is slightly below baseline (0.6174); Instance/Triplet show small mAP gains (~+0.006); if prioritizing Rank@1, Instance is best (0.8007). *(Fixed)*
- [x] **Competitive summary sentence for variants missing** — requirement.md asks for "Summary of baseline performance and your variant's results" as a paragraph, not just a table. *(Fixed)*

---

## Category 2: Opportunities to Add More Detail

- [x] **Domain shift delta quantification** — §4.2 only states the phenomenon. Add explicit deltas: ΔRank@1 = 0.8774 − 0.3299 = **0.5475**, ΔmAP = **0.5519**, to strengthen the argument. *(Fixed)*
- [x] **Improvement proposals too brief** — §7's two proposals are one sentence each. Expand:
  - Re-ranking: cite typical mAP gain of +3%~8% from literature; mention k-reciprocal parameters (k1, k2, λ).
  - Random erasing: already built into the repo (`--erasing_p`), reference the specific switch. *(Fixed)*
- [x] **Backbone vs. loss orthogonality gap noted** — The experiment matrix covers ResNet50 × {softmax, circle, instance, triplet} and {DenseNet, HRNet} × softmax. Mention the incomplete factorial design (e.g., HRNet + triplet untested) to clarify experimental scope. *(Fixed)*

---

## Category 3: Visualizations to Add

- [x] **Success case comparison figure** — milestones.md M6 records two success cases (`show_success1.png`, `show_success2.png`, Top-10 all correct). Adding a **success vs. failure** side-by-side figure makes the model's capability boundary more concrete. *(Fixed)*
- [ ] **Training convergence curve** — milestones.md M3 records the full loss/acc convergence (`train.jpg` already generated). No training curve is referenced in the report. Adding it supports the §3.2 training configuration and `train_all` discussion.

---

## Category 4: Numeric Consistency

- [x] **Precision consistency in tables** — Table 1 & 2 use 4 decimal places (e.g., 0.8774), but milestones.md records 6 decimal places (0.877375). Standardize to 4 decimal places throughout.
- [x] **Instance vs. Triplet mAP distinction** — Table 2 shows both as 0.6232, but milestones.md records 0.623174 vs. 0.623227. These round to the same value; acceptable, but verify no copy-paste error.

---

## Category 5: Extended Content

- [x] **AI Safety section more concrete** — §8 uses abstract policy language. Add a concrete scenario: e.g., if a ReID database used in a shopping mall leaks, it can be combined with purchasing records to reconstruct individual trajectories — linking directly to the "minimize retention" safeguard. *(Fixed)*

---

## Summary Table

| Priority | Item | Status |
|----------|------|--------|
| High | Q3 output dimension added to Quick Questions | Done |
| High | `train_all` impact discussion expanded | Done |
| Medium | Loss variant analysis paragraph | Done |
| Medium | Domain shift delta quantification | Done |
| Medium | Improvement proposals expanded | Done |
| Medium | Success case comparison figure | Done |
| Low | Training convergence curve | Pending |
| Low | Backbone vs. loss orthogonality note | Done |
| Low | AI Safety concrete scenario | Done |
