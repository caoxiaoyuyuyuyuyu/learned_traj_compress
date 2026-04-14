<!-- bootstrap: 10000 resamples, seed=42, generator=scripts/bootstrap_dispersion_ci.py -->
| Backbone | Retrieval | Metric | Split | N | Mean | 95% CI |
|----------|-----------|--------|-------|---|------|--------|
| Qwen-3B | no-API | action_seq_acc | N=8 | 125 | 0.6058 | [0.5950, 0.6162] |
| Qwen-3B | no-API | ROUGE-L | N=8 | 125 | 0.2142 | [0.2031, 0.2251] |
| Llama-1B | no-API | action_seq_acc | N=8 | 125 | 0.6058 | [0.5954, 0.6160] |
| Llama-1B | no-API | ROUGE-L | N=8 | 125 | 0.2124 | [0.2014, 0.2231] |
| Qwen-3B | no-API | em_partial | N2 | 251 | 0.1116 | [0.0837, 0.1394] |
| Qwen-3B | no-API | em_partial | N4 | 131 | 0.0840 | [0.0592, 0.1107] |
| Qwen-3B | no-API | em_partial | N8 | 125 | 0.0250 | [0.0130, 0.0380] |
| Llama-1B | no-API | em_partial | N2 | 251 | 0.0737 | [0.0518, 0.0976] |
| Llama-1B | no-API | em_partial | N4 | 131 | 0.0630 | [0.0420, 0.0859] |
| Llama-1B | no-API | em_partial | N8 | 125 | 0.0140 | [0.0050, 0.0240] |
| Llama-1B | oracle | em_partial | N2 | 251 | 0.4104 | [0.3665, 0.4542] |
| Llama-1B | oracle | em_partial | N4 | 131 | 0.4008 | [0.3569, 0.4447] |
| Llama-1B | oracle | em_partial | N8 | 125 | 0.3510 | [0.3200, 0.3820] |
| Llama-1B | BM25 | em_partial | N2 | 251 | 0.1355 | [0.1056, 0.1673] |
| Llama-1B | BM25 | em_partial | N4 | 131 | 0.0935 | [0.0687, 0.1202] |
| Llama-1B | BM25 | em_partial | N8 | 125 | 0.0660 | [0.0500, 0.0830] |
