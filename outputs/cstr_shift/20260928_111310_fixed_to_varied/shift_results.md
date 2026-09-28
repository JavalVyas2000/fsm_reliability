# CSTR distribution shift: fixed nominal plant → varied nominal plants (first proposals)

Fixed-plant train n=1400, fixed test n=300; varied train n=1389, varied test n=298 (failure prevalence fixed test 0.65, varied test 0.80).

| Family | AUROC fixed→fixed | AUROC fixed→varied [95% CI] | AUROC varied→varied | drop under shift | shift − in-distribution [95% CI] |
|---|---|---|---|---|---|
| context_action | 0.887 | 0.453 [0.374, 0.528] | 0.889 | +0.435 | -0.437 [-0.530, -0.339] |
| context_only_no_action | 0.843 | 0.438 [0.362, 0.515] | 0.862 | +0.405 | -0.424 [-0.528, -0.315] |
| token_confidence | 0.627 | 0.589 [0.516, 0.662] | 0.714 | +0.037 | -0.125 [-0.218, -0.033] |
| attention | 0.672 | 0.476 [0.400, 0.548] | 0.708 | +0.195 | -0.231 [-0.354, -0.109] |
| hidden | 0.695 | 0.582 [0.506, 0.653] | 0.723 | +0.113 | -0.141 [-0.244, -0.035] |
| all_internal | 0.677 | 0.460 [0.384, 0.537] | 0.766 | +0.217 | -0.306 [-0.408, -0.202] |
| context_action+all_internal | 0.849 | 0.407 [0.332, 0.483] | 0.869 | +0.442 | -0.462 [-0.560, -0.357] |

Under shift, context_action + all internals − context_action: -0.046 [-0.075, -0.019]