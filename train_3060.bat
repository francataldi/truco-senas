@echo off
python -m pytest -q || exit /b 1
python scripts\train_ppo.py --config configs\ppo_3060.yaml
python scripts\analyze_signals.py --checkpoint runs\parejas_senas\last.pt
python scripts\train_ppo.py --config configs\ppo_sin_senas.yaml
python scripts\train_ppo.py --config configs\ppo_espias.yaml
python scripts\train_cfr.py --iterations 300000 --workers 5
