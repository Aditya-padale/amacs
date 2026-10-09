# AMACS Fixture Benchmark

All values below come from the recorded offline run. The fixture uses the deterministic stub provider; it is not evidence of quality against a real LLM.

| system | fault rate | adaptive | runs | success rate | recovery rate | mean latency (s) |
|---|---:|:---:|---:|---:|---:|---:|
| single-call | 0.00 | False | 9 | 1.000 | 0.000 | 0.000008 |
| amacs | 0.00 | False | 9 | 1.000 | 0.000 | 0.059296 |
| amacs | 0.00 | True | 9 | 1.000 | 0.000 | 0.014351 |
| amacs | 0.25 | False | 9 | 1.000 | 1.000 | 0.456765 |
| amacs | 0.25 | True | 9 | 1.000 | 1.000 | 0.459288 |
| amacs | 0.50 | False | 9 | 1.000 | 1.000 | 0.900496 |
| amacs | 0.50 | True | 9 | 1.000 | 1.000 | 1.183298 |
