# Customer support routing policy v1

**Primary principle:** Route to the team responsible for the customer's **explicit requested action**, not the underlying cause or incidental keywords. When no action is requested, route by the problem that needs resolution.

- `billing`: charges, refunds, payment, paid-plan activation, subscription cancellation or billing-related changes.
- `technical`: crashes, slow performance, freezing, errors, uploads, malfunctioning product features.
- `account`: sign-in, password recovery, verification, profile and account-detail changes.
- `general`: product information, feature suggestions, feedback, support hours, general questions without an issue requiring another team.

## Decision procedure
1. Extract the action the customer wants support to perform. If several actions are requested, choose the explicit primary request; flag unresolved multi-action requests for review.
2. Route that action to the responsible team. Do not classify based on keywords alone.
3. If no explicit action is stated, route the reported issue to its responsible team.
4. Preserve secondary issues as context; do not silently rewrite golden labels based on model predictions.
5. If the action or ownership remains unclear, mark for human label review rather than treating the model's answer as ground truth.

## Anchors
- TC002: request to **cancel subscription** because of crashes → primary `billing`, secondary `technical`.
- TC012: settings page freezes; user asks if account is wrong → `technical`.
- TC018: password-reset link errors → `account` because recovery is the primary task.
- TC028: account dashboard errors after successful login → `technical`.

**Versioning:** Do not modify datasets v1/v2 or their saved evaluations. Dataset v3 changes labels and metadata under this policy. Compare classifier prompt versions on the *same* dataset and judge rubric. A score change from relabeling alone is not a model regression.
