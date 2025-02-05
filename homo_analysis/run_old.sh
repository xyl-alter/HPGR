python main.py --datasets facebook \
 --eps-list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 \
 --methods init \
 --delta-list 0.1 0.4 0.7 1.0 \
 --graph-seeds 2022 2023 2024 2025 2026 \
 --split-percentage 0.5 0.25 \
 --device cuda:3 \
 --save-root ../results/analysis/facebook/init

 python main.py --datasets facebook \
 --eps-list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 \
 --methods modified \
 --sbm-ratio-list 0.1 0.4 0.7 1.0 \
 --degree-ratio-list 0.0 \
 --graph-seeds 2022 2023 2024 2025 2026 \
 --split-percentage 0.5 0.25 \
 --device cuda:3 \
 --tau 0 0.5 1 5 \
 --save-root ../results/analysis/facebook/modified1

 python main.py --datasets facebook \
 --eps-list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 \
 --methods modified \
 --sbm-ratio-list 0.1 0.4 0.7 1.0 \
 --degree-ratio-list 0.0 \
 --graph-seeds 2022 2023 2024 2025 2026 \
 --split-percentage 0.5 0.25 \
 --device cuda:3 \
 --tau 0.25 0.75 2 10 \
 --save-root ../results/analysis/facebook/modified2 