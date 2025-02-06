Supplementary materials for the submission of papers for the EDBT 2026 conference. 

To reproduce the results, run "sh run.sh".

Parameter interpretation:

1. --dataset: a list of graph classification datasest to be used
2. --gnn-models: a list of GNN models to be tested
3. --eps-list: privacy budget
4. --methods: blink, hpgr or both
5. --variant: sampling strategy
6. --delta-list: proportions of privacy budget used for degree query in Blink method
7. --sbm-ratio-list: proportions of privacy budget used for degree vector query in HPGR method
8. --split-percentage: dataset partition ratio
9. --device: cuda number or cpu
10. --save-root: automatically generated path for saving results
11. --tau: 0 means hard query in HPGR, 1 means soft query. It can also be set as other positive floating point numbers.

All rights reserved. No part of this repository may be used, copied, modified, or distributed without explicit permission from the author.
