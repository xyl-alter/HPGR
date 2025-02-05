python main2.py --datasets cora computers laftfm facebook \
 --gnn-models gcn graphsage gat \
 --eps-list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 \
 --methods blink hpgr \
 --variant soft hard  \
 --sbm-ratio-list 0.1 0.4 0.7 0.9 \
 --split-percentage 0.5 0.25 \
 --device cuda:0 \
 --save-root ./results/ 

