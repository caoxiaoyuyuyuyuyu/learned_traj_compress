| Backbone | Retrieval | Metric | Split | N | Mean | 95% CI |
|----------|-----------|--------|-------|---|------|--------|
| Qwen-3B | no-API | action_seq_acc | N=8 | 125 | 0.6058 | [0.5953, 0.6162] |
| Qwen-3B | no-API | ROUGE-L | N=8 | 125 | 0.2142 | [0.2037, 0.2244] |
| Llama-1B | no-API | action_seq_acc | N=8 | 125 | 0.6058 | [0.5946, 0.6158] |
| Llama-1B | no-API | ROUGE-L | N=8 | 125 | 0.2124 | [0.2023, 0.2237] |
| Qwen-3B | no-API | em_partial | N2 | 251 | 0.1116 | [0.0857, 0.1394] |
| Qwen-3B | no-API | em_partial | N4 | 131 | 0.0840 | [0.0611, 0.1107] |
| Qwen-3B | no-API | em_partial | N8 | 125 | 0.0250 | [0.0130, 0.0380] |
| Llama-1B | no-API | em_partial | N2 | 251 | 0.0737 | [0.0518, 0.0956] |
| Llama-1B | no-API | em_partial | N4 | 131 | 0.0630 | [0.0420, 0.0859] |
| Llama-1B | no-API | em_partial | N8 | 125 | 0.0140 | [0.0050, 0.0250] |
| Llama-1B | oracle | em_partial | N2 | 251 | 0.4104 | [0.3645, 0.4522] |
| Llama-1B | oracle | em_partial | N4 | 131 | 0.4008 | [0.3569, 0.4447] |
| Llama-1B | oracle | em_partial | N8 | 125 | 0.3510 | [0.3180, 0.3840] |
| Llama-1B | BM25 | em_partial | N2 | 251 | 0.1355 | [0.1076, 0.1653] |
| Llama-1B | BM25 | em_partial | N4 | 131 | 0.0935 | [0.0687, 0.1221] |
| Llama-1B | BM25 | em_partial | N8 | 125 | 0.0660 | [0.0480, 0.0830] |
