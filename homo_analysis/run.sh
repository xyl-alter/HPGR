

# python main.py --datasets cora \
#  --eps-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
#  --methods modified \
#  --delta-list 0.1 \
#  --sbm-ratio-list 0.1 \
#  --degree-ratio-list 0.0 \
#  --graph-seeds 2022 2023 2024 2025 2026 \
#  --split-percentage 0.5 0.25 \
#  --device cuda:0 \
#  --tau 0 1 \
#  --save-root ../results/analysis_v5/prior/cora \
#  --return-prior &

python main.py --datasets lastfm \
 --eps-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
 --methods modified \
 --delta-list 0.1 \
 --sbm-ratio-list 0.1 \
 --degree-ratio-list 0.0 \
 --graph-seeds 2022 2023 2024 2025 2026 \
 --split-percentage 0.5 0.25 \
 --device cuda:1 \
 --tau 0 1 \
 --save-root ../results/analysis_v5/prior/lastfm \
 --return-prior &

python main.py --datasets computers \
 --eps-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
 --methods modified \
 --delta-list 0.1 \
 --sbm-ratio-list 0.1 \
 --degree-ratio-list 0.0 \
 --graph-seeds 2022 2023 2024 2025 2026 \
 --split-percentage 0.5 0.25 \
 --device cuda:2 \
 --tau 0 1 \
 --save-root ../results/analysis_v5/prior/computers \
 --return-prior &

python main.py --datasets facebook \
 --eps-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
 --methods modified \
 --delta-list 0.1 \
 --sbm-ratio-list 0.1 \
 --degree-ratio-list 0.0 \
 --graph-seeds 2022 2023 2024 2025 2026 \
 --split-percentage 0.5 0.25 \
 --device cuda:3 \
 --tau 0 1 \
 --save-root ../results/analysis_v5/prior/facebook \
 --return-prior &
