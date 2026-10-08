
## Stage-1 predictability on test tasks, sopbench

| signal | scheme | R2 | Spearman | Spearman within domain |
|---|---|---|---|---|
| mean_asst_msgs | new_domain | +0.274 | +0.584 | +0.637 |
| mean_asst_msgs | new_procedures | +0.355 | +0.595 | +0.625 |
| mean_tool_calls | new_domain | +0.375 | +0.627 | +0.617 |
| mean_tool_calls | new_procedures | +0.406 | +0.634 | +0.600 |
| share_action_correct | new_domain | +0.046 | +0.293 | +0.205 |
| share_action_correct | new_procedures | +0.043 | +0.285 | +0.173 |
| share_called_target | new_domain | +0.019 | +0.046 | +0.096 |
| share_called_target | new_procedures | +0.046 | +0.161 | +0.045 |
| share_dirgraph | new_domain | +0.069 | +0.294 | +0.151 |
| share_dirgraph | new_procedures | +0.128 | +0.381 | +0.135 |
| share_zero_tool | new_domain | +0.171 | +0.454 | +0.384 |
| share_zero_tool | new_procedures | +0.236 | +0.489 | +0.341 |

## Spearman of each privileged signal with fitted beta, inside domains, sopbench

| signal | mean within-domain rho |
|---|---|
| share_called_target | +0.659 |
| share_dirgraph | -0.638 |
| share_action_correct | -0.503 |
| mean_tool_calls | +0.331 |
| mean_asst_msgs | +0.271 |
| share_zero_tool | +0.130 |

(for SOPBench, within allow/refuse label cells:)
- mean_tool_calls: +0.154
- mean_asst_msgs: +0.199
- share_called_target: +0.214
- share_zero_tool: +0.249
- share_dirgraph: -0.521
- share_action_correct: -0.704

## P(refuse) AUC against the label, sopbench

| scheme | logistic pi pooled | logistic pi within domain | rubric p_refusal pooled | rubric within domain |
|---|---|---|---|---|
| new_domain | 0.581 | 0.593 | 0.606 | 0.571 |
| new_procedures | 0.605 | 0.594 | 0.606 | 0.571 |

label share positive (refuse): 0.653, n=830

## Stage-1 predictability on test tasks, tau2

| signal | scheme | R2 | Spearman | Spearman within domain |
|---|---|---|---|---|
| mean_msgs | new_domain | +0.164 | +0.633 | +0.069 |
| mean_msgs | new_procedures | +0.451 | +0.661 | +0.058 |
| mean_tool_calls | new_domain | -0.038 | +0.103 | +0.133 |
| mean_tool_calls | new_procedures | +0.056 | +0.241 | +0.226 |
| share_transfer | new_domain | -0.294 | -0.107 | +0.140 |
| share_transfer | new_procedures | +0.072 | +0.432 | +0.131 |
| term_infrastructure_error | new_domain | -0.084 | +0.024 | +0.158 |
| term_infrastructure_error | new_procedures | -0.053 | +0.071 | +0.009 |
| term_max_steps | new_domain | +0.022 | +0.201 | -0.126 |
| term_max_steps | new_procedures | +0.035 | +0.258 | -0.170 |
| term_too_many_errors | new_domain | +0.008 | +0.197 | +nan |
| term_too_many_errors | new_procedures | +0.046 | +0.257 | +0.004 |
| term_user_stop | new_domain | -0.036 | +0.126 | +0.122 |
| term_user_stop | new_procedures | -0.033 | +0.091 | -0.002 |

## Spearman of each privileged signal with fitted beta, inside domains, tau2

| signal | mean within-domain rho |
|---|---|
| mean_msgs | +0.661 |
| mean_tool_calls | +0.436 |
| term_infrastructure_error | +0.303 |
| term_too_many_errors | +0.286 |
| term_user_stop | -0.268 |
| share_transfer | +0.063 |
| term_max_steps | +0.038 |

## P(refuse) AUC against the label, tau2

| scheme | logistic pi pooled | logistic pi within domain | rubric p_refusal pooled | rubric within domain |
|---|---|---|---|---|
| new_domain | 0.378 | 0.559 | 0.599 | 0.676 |
| new_procedures | 0.209 | 0.355 | 0.599 | 0.676 |

label share positive (refuse_or_transfer): 0.122, n=278