@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call scripts\buscar_python.bat
if not defined PY (pause & exit /b 1)
"%PY%" -m pytest -q || (pause & exit /b 1)
set L=runs_largo
if not exist %L% mkdir %L%
set DESPIERTA=1
start "Manteniendo la PC despierta" /min cmd /c mantener_despierta.bat --hasta %L%\listo.txt --horas 48
title Entrenamiento largo 1/4: mano a mano
"%PY%" scripts\train_ppo.py --config configs\ppo_duelo.yaml
title Entrenamiento largo 2/4: con senas
"%PY%" scripts\train_ppo.py --config configs\ppo_principal_largo.yaml
title Entrenamiento largo 3/4: sin senas
"%PY%" scripts\train_ppo.py --config configs\ppo_sin_senas_largo.yaml
title Entrenamiento largo 4/4: con espias
"%PY%" scripts\train_ppo.py --config configs\ppo_espias_largo.yaml
title Analisis del entrenamiento largo
set C=%L%\comparaciones
if not exist %L%\parejas_senas\senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %L%\parejas_senas\last.pt
if not exist %L%\parejas_espias\senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %L%\parejas_espias\last.pt
if not exist %C%\duelo_vs_cfr.json "%PY%" scripts\comparar.py --a ppo:%L%\duelo\last.pt --b cfr:runs\cfr\heads_up.npz --players 2 --matches 2000 --workers 5 --json %C%\duelo_vs_cfr.json
if not exist %C%\duelo_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\duelo\last.pt --players 2 --matches 2000 --workers 5 --json %C%\duelo_vs_clasico.json
if not exist %C%\senas_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_senas\last.pt --matches 2000 --workers 5 --json %C%\senas_vs_clasico.json
if not exist %C%\sin_senas_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_sin_senas\last.pt --matches 2000 --workers 5 --json %C%\sin_senas_vs_clasico.json
if not exist %C%\espias_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_espias\last.pt --matches 2000 --workers 5 --json %C%\espias_vs_clasico.json
if not exist %C%\senas_largo_vs_corto.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_senas\last.pt --b ppo:runs\parejas_senas\last.pt --matches 2000 --workers 5 --json %C%\senas_largo_vs_corto.json
if not exist %L%\liga\liga.json "%PY%" scripts\league.py --runs runs\parejas_senas runs\parejas_sin_senas runs\parejas_espias %L%\parejas_senas %L%\parejas_sin_senas %L%\parejas_espias --per-run 1 --matches 400 --out %L%\liga
if not exist %L%\liga_duelo\liga.json "%PY%" scripts\league.py --players 2 --specs cfr:runs\cfr\heads_up.npz ppo:%L%\duelo\last.pt --matches 400 --out %L%\liga_duelo
echo listo> %L%\listo.txt
title Entrenamiento largo terminado
pause
