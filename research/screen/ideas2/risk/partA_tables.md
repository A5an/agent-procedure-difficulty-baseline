
### overall_all28

| should | correct | unsafe | wrongful_refusal | wrong_execution | process_only | empty | n |
|---|---|---|---|---|---|---|---|
| 0 | 0.521 | 0.416 | 0.000 | 0.000 | 0.035 | 0.029 | 15172.000 |
| 1 | 0.307 | 0.000 | 0.142 | 0.156 | 0.374 | 0.021 | 8063.000 |
| all | 0.446 | 0.272 | 0.050 | 0.054 | 0.152 | 0.026 | 23235.000 |

### domain_all28

| domain | correct | unsafe | wrongful_refusal | wrong_execution | process_only | empty |
|---|---|---|---|---|---|---|
| bank | 0.503 | 0.206 | 0.088 | 0.094 | 0.088 | 0.021 |
| dmv | 0.616 | 0.158 | 0.054 | 0.061 | 0.096 | 0.014 |
| healthcare | 0.528 | 0.239 | 0.044 | 0.034 | 0.125 | 0.029 |
| hotel | 0.287 | 0.346 | 0.045 | 0.075 | 0.205 | 0.043 |
| library | 0.318 | 0.354 | 0.061 | 0.065 | 0.181 | 0.020 |
| online_market | 0.451 | 0.272 | 0.030 | 0.019 | 0.205 | 0.023 |
| university | 0.557 | 0.363 | 0.014 | 0.002 | 0.058 | 0.006 |

### overall_no_empty3

| should | correct | unsafe | wrongful_refusal | wrong_execution | process_only | empty | n |
|---|---|---|---|---|---|---|---|
| 0 | 0.502 | 0.456 | 0.000 | 0.000 | 0.036 | 0.007 | 13546.000 |
| 1 | 0.302 | 0.000 | 0.134 | 0.160 | 0.401 | 0.003 | 7199.000 |
| all | 0.432 | 0.297 | 0.046 | 0.056 | 0.163 | 0.006 | 20745.000 |

### domain_no_empty3

| domain | correct | unsafe | wrongful_refusal | wrong_execution | process_only | empty |
|---|---|---|---|---|---|---|
| bank | 0.495 | 0.221 | 0.088 | 0.099 | 0.092 | 0.006 |
| dmv | 0.602 | 0.174 | 0.058 | 0.059 | 0.106 | 0.002 |
| healthcare | 0.512 | 0.266 | 0.041 | 0.035 | 0.139 | 0.007 |
| hotel | 0.270 | 0.380 | 0.039 | 0.079 | 0.221 | 0.011 |
| library | 0.308 | 0.376 | 0.058 | 0.063 | 0.193 | 0.002 |
| online_market | 0.440 | 0.299 | 0.026 | 0.019 | 0.214 | 0.003 |
| university | 0.520 | 0.405 | 0.010 | 0.002 | 0.064 | 0.000 |

### flagged_configs

| config | correct | unsafe | wrongful_refusal | wrong_execution | process_only | empty |
|---|---|---|---|---|---|---|
| gemini-2.5-pro-mode_fc | 0.581 | 0.092 | 0.065 | 0.039 | 0.125 | 0.099 |
| gpt-5-mini-mode_fc | 0.681 | 0.059 | 0.081 | 0.057 | 0.057 | 0.066 |
| gpt-5-mode_fc | 0.433 | 0.022 | 0.076 | 0.031 | 0.013 | 0.425 |

### per_config

| config | correct | unsafe | wrongful_refusal | wrong_execution | process_only | empty |
|---|---|---|---|---|---|---|
| claude-3-5-sonnet-20241022-mode_act-only | 0.147 | 0.494 | 0.022 | 0.046 | 0.292 | 0.000 |
| claude-3-5-sonnet-20241022-mode_fc | 0.376 | 0.336 | 0.031 | 0.047 | 0.208 | 0.001 |
| claude-3-5-sonnet-20241022-mode_react | 0.289 | 0.377 | 0.036 | 0.042 | 0.248 | 0.007 |
| claude-3-7-sonnet-20250219-mode_fc | 0.523 | 0.284 | 0.036 | 0.053 | 0.104 | 0.000 |
| claude-3-7-sonnet-20250219-thinking-mode_fc | 0.502 | 0.293 | 0.037 | 0.049 | 0.118 | 0.000 |
| deepseek-r1-mode_react | 0.555 | 0.106 | 0.065 | 0.035 | 0.174 | 0.064 |
| gemini-1.5-pro-mode_fc | 0.304 | 0.408 | 0.028 | 0.041 | 0.219 | 0.000 |
| gemini-2.0-flash-001-mode_fc | 0.306 | 0.402 | 0.037 | 0.034 | 0.220 | 0.000 |
| gemini-2.0-flash-thinking-exp-mode_react | 0.652 | 0.135 | 0.046 | 0.030 | 0.110 | 0.028 |
| gemini-2.5-flash-mode_fc | 0.681 | 0.154 | 0.070 | 0.040 | 0.055 | 0.000 |
| gemini-2.5-pro-mode_fc | 0.581 | 0.092 | 0.065 | 0.039 | 0.125 | 0.099 |
| gemma-4-e2b-it-mode_fc | 0.257 | 0.389 | 0.018 | 0.076 | 0.245 | 0.016 |
| gemma-4-e4b-it-mode_fc | 0.558 | 0.199 | 0.036 | 0.053 | 0.153 | 0.001 |
| gpt-4.1-mini-mode_fc | 0.447 | 0.322 | 0.046 | 0.046 | 0.140 | 0.000 |
| gpt-4.1-mode_fc | 0.654 | 0.200 | 0.042 | 0.048 | 0.048 | 0.007 |
| gpt-4o-mini-mode_fc | 0.416 | 0.308 | 0.075 | 0.066 | 0.134 | 0.001 |
| gpt-4o-mode_fc | 0.601 | 0.213 | 0.063 | 0.024 | 0.099 | 0.000 |
| gpt-5-mini-mode_fc | 0.681 | 0.059 | 0.081 | 0.057 | 0.057 | 0.066 |
| gpt-5-mode_fc | 0.433 | 0.022 | 0.076 | 0.031 | 0.013 | 0.425 |
| llama3.1-70b-instruct-mode_react | 0.387 | 0.316 | 0.077 | 0.045 | 0.176 | 0.000 |
| llama3.1-8b-instruct-mode_react | 0.146 | 0.466 | 0.054 | 0.154 | 0.180 | 0.000 |
| o4-mini-high-mode_fc | 0.742 | 0.088 | 0.065 | 0.042 | 0.048 | 0.014 |
| qwen2.5-14b-instruct-mode_react | 0.308 | 0.386 | 0.043 | 0.057 | 0.206 | 0.000 |
| qwen2.5-32b-instruct-mode_react | 0.372 | 0.319 | 0.057 | 0.054 | 0.198 | 0.000 |
| qwen2.5-72b-instruct-mode_react | 0.339 | 0.317 | 0.064 | 0.051 | 0.230 | 0.000 |
| qwen2.5-7b-instruct-mode_react | 0.090 | 0.546 | 0.022 | 0.093 | 0.249 | 0.000 |
| qwen3.5-2b-mode_fc | 0.466 | 0.242 | 0.055 | 0.107 | 0.129 | 0.000 |
| qwen3.5-4b-mode_fc | 0.687 | 0.134 | 0.037 | 0.058 | 0.084 | 0.000 |

### var_all28

| index | rate | n_cases | n_procs | eta2_domain_adj | eta2_proc | eta2_proc_adj | eta2_proc_chance | mean_rate |
|---|---|---|---|---|---|---|---|---|
| 0 | unsafe | 542 | 66 | 0.161 | 0.335 | 0.244 | 0.119 | 0.416 |
| 1 | wrongful_refusal | 288 | 70 | 0.064 | 0.587 | 0.456 | 0.245 | 0.143 |
| 2 | wrong_execution | 288 | 70 | 0.080 | 0.872 | 0.832 | 0.237 | 0.156 |
| 3 | process_only | 830 | 70 | 0.050 | 0.117 | 0.036 | 0.082 | 0.152 |
| 4 | fail | 830 | 70 | 0.167 | 0.285 | 0.220 | 0.084 | 0.535 |

### var_no_empty3

| index | rate | n_cases | n_procs | eta2_domain_adj | eta2_proc | eta2_proc_adj | eta2_proc_chance | mean_rate |
|---|---|---|---|---|---|---|---|---|
| 0 | unsafe | 542 | 66 | 0.163 | 0.337 | 0.247 | 0.119 | 0.456 |
| 1 | wrongful_refusal | 288 | 70 | 0.072 | 0.589 | 0.459 | 0.246 | 0.134 |
| 2 | wrong_execution | 288 | 70 | 0.088 | 0.845 | 0.795 | 0.237 | 0.160 |
| 3 | process_only | 830 | 70 | 0.047 | 0.112 | 0.031 | 0.082 | 0.163 |
| 4 | fail | 830 | 70 | 0.175 | 0.295 | 0.231 | 0.084 | 0.563 |

### split_half

| index | rate | n_procs | n_agents | half_r | half_r_lo | half_r_hi | spearman_brown | agents | min_cases |
|---|---|---|---|---|---|---|---|---|---|
| 0 | unsafe | 48 | 28 | 0.867 | 0.793 | 0.926 | 0.929 | all28 | 3 |
| 1 | unsafe | 26 | 28 | 0.886 | 0.784 | 0.955 | 0.940 | all28 | 6 |
| 2 | wrongful_refusal | 28 | 28 | 0.828 | 0.676 | 0.927 | 0.906 | all28 | 3 |
| 3 | wrongful_refusal | 14 | 28 | 0.818 | 0.591 | 0.951 | 0.900 | all28 | 6 |
| 4 | wrong_execution | 28 | 28 | 0.815 | 0.747 | 0.877 | 0.898 | all28 | 3 |
| 5 | wrong_execution | 14 | 28 | 0.749 | 0.567 | 0.893 | 0.856 | all28 | 6 |
| 6 | process_only | 54 | 28 | 0.901 | 0.838 | 0.945 | 0.948 | all28 | 3 |
| 7 | process_only | 37 | 28 | 0.891 | 0.818 | 0.945 | 0.942 | all28 | 6 |
| 8 | fail | 54 | 28 | 0.873 | 0.752 | 0.936 | 0.932 | all28 | 3 |
| 9 | fail | 37 | 28 | 0.900 | 0.832 | 0.952 | 0.947 | all28 | 6 |
| 10 | unsafe | 48 | 25 | 0.860 | 0.782 | 0.926 | 0.925 | no_empty3 | 3 |
| 11 | unsafe | 26 | 25 | 0.881 | 0.776 | 0.951 | 0.937 | no_empty3 | 6 |
| 12 | wrongful_refusal | 28 | 25 | 0.843 | 0.717 | 0.932 | 0.915 | no_empty3 | 3 |
| 13 | wrongful_refusal | 14 | 25 | 0.819 | 0.631 | 0.943 | 0.901 | no_empty3 | 6 |
| 14 | wrong_execution | 28 | 25 | 0.817 | 0.748 | 0.881 | 0.899 | no_empty3 | 3 |
| 15 | wrong_execution | 14 | 25 | 0.750 | 0.553 | 0.880 | 0.857 | no_empty3 | 6 |
| 16 | process_only | 54 | 25 | 0.894 | 0.834 | 0.940 | 0.944 | no_empty3 | 3 |
| 17 | process_only | 37 | 25 | 0.886 | 0.812 | 0.940 | 0.939 | no_empty3 | 6 |
| 18 | fail | 54 | 25 | 0.860 | 0.751 | 0.927 | 0.925 | no_empty3 | 3 |
| 19 | fail | 37 | 25 | 0.886 | 0.803 | 0.943 | 0.939 | no_empty3 | 6 |

### a3_all28

| index | method | rate | rho | lo | hi | n_procs |
|---|---|---|---|---|---|---|
| 0 | adele | fail | 0.226 | -0.038 | 0.457 | 70 |
| 1 | adele | unsafe | 0.421 | 0.052 | 0.676 | 48 |
| 2 | adele | wrongful_refusal | 0.041 | -0.529 | 0.541 | 28 |
| 3 | adele | wrong_execution | -0.104 | -0.617 | 0.505 | 28 |
| 4 | adele | process_only | 0.116 | -0.193 | 0.409 | 54 |
| 5 | adele | share_perform | -0.266 | -0.482 | -0.010 | 70 |
| 6 | lad2_adele | fail | 0.411 | 0.158 | 0.612 | 70 |
| 7 | lad2_adele | unsafe | 0.287 | -0.105 | 0.538 | 48 |
| 8 | lad2_adele | wrongful_refusal | 0.326 | -0.253 | 0.803 | 28 |
| 9 | lad2_adele | wrong_execution | -0.005 | -0.566 | 0.520 | 28 |
| 10 | lad2_adele | process_only | 0.315 | -0.033 | 0.604 | 54 |
| 11 | lad2_adele | share_perform | -0.034 | -0.290 | 0.199 | 70 |
| 12 | cat_proc_adele | fail | 0.355 | 0.099 | 0.562 | 70 |
| 13 | cat_proc_adele | unsafe | 0.336 | -0.057 | 0.560 | 48 |
| 14 | cat_proc_adele | wrongful_refusal | 0.164 | -0.408 | 0.678 | 28 |
| 15 | cat_proc_adele | wrong_execution | -0.115 | -0.677 | 0.480 | 28 |
| 16 | cat_proc_adele | process_only | 0.277 | -0.059 | 0.580 | 54 |
| 17 | cat_proc_adele | share_perform | -0.112 | -0.357 | 0.155 | 70 |
| 18 | grp_proc_adele | fail | 0.453 | 0.177 | 0.639 | 70 |
| 19 | grp_proc_adele | unsafe | 0.264 | -0.143 | 0.527 | 48 |
| 20 | grp_proc_adele | wrongful_refusal | 0.284 | -0.287 | 0.773 | 28 |
| 21 | grp_proc_adele | wrong_execution | -0.017 | -0.601 | 0.526 | 28 |
| 22 | grp_proc_adele | process_only | 0.215 | -0.112 | 0.565 | 54 |
| 23 | grp_proc_adele | share_perform | 0.007 | -0.230 | 0.233 | 70 |

### a3_diff_all28

| index | method | contrast | diff | lo | hi |
|---|---|---|---|---|---|
| 0 | adele | unsafe minus fail | 0.195 | -0.156 | 0.472 |
| 1 | adele | wrongful_refusal minus fail | -0.186 | -0.762 | 0.344 |
| 2 | adele | wrong_execution minus fail | -0.330 | -0.822 | 0.261 |
| 3 | adele | unsafe minus wrongful_refusal | 0.381 | -0.299 | 1.000 |
| 4 | lad2_adele | unsafe minus fail | -0.124 | -0.463 | 0.169 |
| 5 | lad2_adele | wrongful_refusal minus fail | -0.085 | -0.700 | 0.465 |
| 6 | lad2_adele | wrong_execution minus fail | -0.416 | -1.011 | 0.188 |
| 7 | lad2_adele | unsafe minus wrongful_refusal | -0.039 | -0.741 | 0.656 |
| 8 | cat_proc_adele | unsafe minus fail | -0.020 | -0.378 | 0.225 |
| 9 | cat_proc_adele | wrongful_refusal minus fail | -0.192 | -0.796 | 0.377 |
| 10 | cat_proc_adele | wrong_execution minus fail | -0.471 | -1.027 | 0.162 |
| 11 | cat_proc_adele | unsafe minus wrongful_refusal | 0.172 | -0.500 | 0.835 |
| 12 | grp_proc_adele | unsafe minus fail | -0.189 | -0.535 | 0.090 |
| 13 | grp_proc_adele | wrongful_refusal minus fail | -0.168 | -0.773 | 0.382 |
| 14 | grp_proc_adele | wrong_execution minus fail | -0.469 | -1.044 | 0.092 |
| 15 | grp_proc_adele | unsafe minus wrongful_refusal | -0.020 | -0.739 | 0.669 |

### a3_no_empty3

| index | method | rate | rho | lo | hi | n_procs |
|---|---|---|---|---|---|---|
| 0 | adele | fail | 0.241 | -0.014 | 0.458 | 70 |
| 1 | adele | unsafe | 0.426 | 0.069 | 0.694 | 48 |
| 2 | adele | wrongful_refusal | 0.053 | -0.517 | 0.568 | 28 |
| 3 | adele | wrong_execution | -0.104 | -0.617 | 0.505 | 28 |
| 4 | adele | process_only | 0.112 | -0.196 | 0.414 | 54 |
| 5 | adele | share_perform | -0.266 | -0.482 | -0.010 | 70 |
| 6 | lad2_adele | fail | 0.398 | 0.139 | 0.596 | 70 |
| 7 | lad2_adele | unsafe | 0.272 | -0.125 | 0.511 | 48 |
| 8 | lad2_adele | wrongful_refusal | 0.357 | -0.220 | 0.834 | 28 |
| 9 | lad2_adele | wrong_execution | -0.005 | -0.566 | 0.520 | 28 |
| 10 | lad2_adele | process_only | 0.298 | -0.040 | 0.604 | 54 |
| 11 | lad2_adele | share_perform | -0.034 | -0.290 | 0.199 | 70 |
| 12 | cat_proc_adele | fail | 0.350 | 0.100 | 0.554 | 70 |
| 13 | cat_proc_adele | unsafe | 0.336 | -0.030 | 0.552 | 48 |
| 14 | cat_proc_adele | wrongful_refusal | 0.214 | -0.367 | 0.717 | 28 |
| 15 | cat_proc_adele | wrong_execution | -0.115 | -0.677 | 0.480 | 28 |
| 16 | cat_proc_adele | process_only | 0.261 | -0.070 | 0.559 | 54 |
| 17 | cat_proc_adele | share_perform | -0.112 | -0.357 | 0.155 | 70 |
| 18 | grp_proc_adele | fail | 0.429 | 0.169 | 0.614 | 70 |
| 19 | grp_proc_adele | unsafe | 0.234 | -0.169 | 0.501 | 48 |
| 20 | grp_proc_adele | wrongful_refusal | 0.336 | -0.237 | 0.800 | 28 |
| 21 | grp_proc_adele | wrong_execution | -0.017 | -0.601 | 0.526 | 28 |
| 22 | grp_proc_adele | process_only | 0.210 | -0.124 | 0.563 | 54 |
| 23 | grp_proc_adele | share_perform | 0.007 | -0.230 | 0.233 | 70 |

### a3_diff_no_empty3

| index | method | contrast | diff | lo | hi |
|---|---|---|---|---|---|
| 0 | adele | unsafe minus fail | 0.185 | -0.170 | 0.481 |
| 1 | adele | wrongful_refusal minus fail | -0.188 | -0.775 | 0.384 |
| 2 | adele | wrong_execution minus fail | -0.345 | -0.834 | 0.238 |
| 3 | adele | unsafe minus wrongful_refusal | 0.373 | -0.305 | 1.019 |
| 4 | lad2_adele | unsafe minus fail | -0.126 | -0.473 | 0.149 |
| 5 | lad2_adele | wrongful_refusal minus fail | -0.041 | -0.641 | 0.535 |
| 6 | lad2_adele | wrong_execution minus fail | -0.402 | -0.974 | 0.217 |
| 7 | lad2_adele | unsafe minus wrongful_refusal | -0.085 | -0.800 | 0.606 |
| 8 | cat_proc_adele | unsafe minus fail | -0.014 | -0.378 | 0.242 |
| 9 | cat_proc_adele | wrongful_refusal minus fail | -0.136 | -0.722 | 0.437 |
| 10 | cat_proc_adele | wrong_execution minus fail | -0.465 | -1.014 | 0.166 |
| 11 | cat_proc_adele | unsafe minus wrongful_refusal | 0.122 | -0.562 | 0.769 |
| 12 | grp_proc_adele | unsafe minus fail | -0.195 | -0.539 | 0.088 |
| 13 | grp_proc_adele | wrongful_refusal minus fail | -0.093 | -0.709 | 0.436 |
| 14 | grp_proc_adele | wrong_execution minus fail | -0.446 | -1.022 | 0.119 |
| 15 | grp_proc_adele | unsafe minus wrongful_refusal | -0.102 | -0.811 | 0.634 |

### rate_corr_no_empty3

| index | a | b | rho_within_domain |
|---|---|---|---|
| 0 | unsafe | wrongful_refusal | -0.372 |
| 1 | unsafe | wrong_execution | -0.085 |
| 2 | wrong_execution | wrongful_refusal | 0.146 |
| 3 | process_only | unsafe | -0.109 |
| 4 | process_only | wrongful_refusal | -0.014 |
| 5 | process_only | wrong_execution | -0.467 |
| 6 | process_only | share_perform | 0.223 |
| 7 | fail | unsafe | 0.726 |
| 8 | fail | wrongful_refusal | 0.069 |
| 9 | fail | wrong_execution | 0.350 |
| 10 | fail | process_only | -0.027 |
| 11 | fail | share_perform | 0.174 |
| 12 | share_perform | unsafe | -0.182 |
| 13 | share_perform | wrongful_refusal | -0.154 |
| 14 | share_perform | wrong_execution | 0.072 |
