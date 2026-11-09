@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call scripts\buscar_python.bat
if not defined PY (pause & exit /b 1)
"%PY%" -m pytest -q || (pause & exit /b 1)
set L=runs_largo
set C=%L%\comparaciones
set F=%L%\figuras
if not exist %L% mkdir %L%
set DESPIERTA=1
start "Manteniendo la PC despierta" /min cmd /c mantener_despierta.bat --hasta %L%\listo.txt --horas 60
title 1/9 Entrenando la IA de mano a mano
"%PY%" scripts\train_ppo.py --config configs\ppo_duelo.yaml
title 2/9 Entrenando la IA con senas
"%PY%" scripts\train_ppo.py --config configs\ppo_principal_largo.yaml
title 3/9 Entrenando la IA sin senas
"%PY%" scripts\train_ppo.py --config configs\ppo_sin_senas_largo.yaml
title 4/9 Entrenando la IA con espias
"%PY%" scripts\train_ppo.py --config configs\ppo_espias_largo.yaml
title 5/9 Entrenando el bot de teoria de juegos
if not exist %L%\cfr\heads_up.npz "%PY%" scripts\train_cfr.py --iterations 5000000 --workers 5 --out %L%\cfr\heads_up.npz
title 6/9 Analizando los idiomas de senas
if not exist %L%\parejas_senas\senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %L%\parejas_senas\last.pt
if not exist %L%\parejas_espias\senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %L%\parejas_espias\last.pt
if not exist %L%\marcador_sorteado\parejas_senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %L%\parejas_senas\last.pt --zero-score-prob 0.3 --out %L%\marcador_sorteado\parejas_senas
title 7/9 Midiendo contra el jugador clasico
if not exist %L%\parejas_senas\vs_heuristico.json "%PY%" scripts\evaluate.py --a ppo:%L%\parejas_senas\last.pt --matches 2000 --duplicado --json %L%\parejas_senas\vs_heuristico.json
if not exist %L%\parejas_espias\vs_heuristico.json "%PY%" scripts\evaluate.py --a ppo:%L%\parejas_espias\last.pt --matches 2000 --duplicado --json %L%\parejas_espias\vs_heuristico.json
if not exist %L%\parejas_sin_senas\vs_heuristico.json "%PY%" scripts\evaluate.py --a ppo:%L%\parejas_sin_senas\last.pt --no-signals --matches 2000 --duplicado --json %L%\parejas_sin_senas\vs_heuristico.json
if not exist %L%\duelo\vs_heuristico.json "%PY%" scripts\evaluate.py --a ppo:%L%\duelo\last.pt --players 2 --matches 2000 --duplicado --json %L%\duelo\vs_heuristico.json
if not exist %L%\cfr\heads_up_vs_heuristico.json "%PY%" scripts\evaluate.py --a cfr:%L%\cfr\heads_up.npz --players 2 --matches 2000 --duplicado --json %L%\cfr\heads_up_vs_heuristico.json
title 8/9 Investigando por que gana cada bot
if not exist %L%\ablaciones\parejas_senas\ablaciones.json "%PY%" scripts\ablations.py --checkpoint %L%\parejas_senas\last.pt --matches 1000 --duplicado --out %L%\ablaciones\parejas_senas
if not exist %L%\ablaciones\parejas_espias\ablaciones.json "%PY%" scripts\ablations.py --checkpoint %L%\parejas_espias\last.pt --matches 1000 --duplicado --out %L%\ablaciones\parejas_espias
if not exist %L%\ablaciones_vs_clasico\parejas_senas\ablaciones.json "%PY%" scripts\ablations.py --checkpoint %L%\parejas_senas\last.pt --opponent heuristic --spy 0.2 --matches 1000 --duplicado --out %L%\ablaciones_vs_clasico\parejas_senas
if not exist %L%\ablaciones_vs_clasico\parejas_espias\ablaciones.json "%PY%" scripts\ablations.py --checkpoint %L%\parejas_espias\last.pt --opponent heuristic --spy 0.2 --matches 1000 --duplicado --out %L%\ablaciones_vs_clasico\parejas_espias
if not exist %C%\senas_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_senas\last.pt --matches 2000 --workers 5 --json %C%\senas_vs_clasico.json
if not exist %C%\sin_senas_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_sin_senas\last.pt --matches 2000 --workers 5 --json %C%\sin_senas_vs_clasico.json
if not exist %C%\espias_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_espias\last.pt --matches 2000 --workers 5 --json %C%\espias_vs_clasico.json
if not exist %C%\senas_sin_espiar_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_senas\last.pt --ruido "{\"blind_a\": true}" --matches 2000 --workers 5 --json %C%\senas_sin_espiar_vs_clasico.json
if not exist %C%\senas_vs_sin_senas.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_senas\last.pt --b ppo:%L%\parejas_sin_senas\last.pt --matches 2000 --workers 5 --json %C%\senas_vs_sin_senas.json
if not exist %C%\senas_vs_espias.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_senas\last.pt --b ppo:%L%\parejas_espias\last.pt --matches 2000 --workers 5 --json %C%\senas_vs_espias.json
if not exist %C%\senas_largo_vs_corto.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_senas\last.pt --b ppo:runs\parejas_senas\last.pt --matches 2000 --workers 5 --json %C%\senas_largo_vs_corto.json
if not exist %C%\sin_senas_largo_vs_corto.json "%PY%" scripts\comparar.py --a ppo:%L%\parejas_sin_senas\last.pt --b ppo:runs\parejas_sin_senas\last.pt --matches 2000 --workers 5 --json %C%\sin_senas_largo_vs_corto.json
if not exist %C%\clasico_vs_clasico_mudo.json "%PY%" scripts\comparar.py --a heuristic --b heuristic-mudo --matches 2000 --workers 5 --json %C%\clasico_vs_clasico_mudo.json
if not exist %C%\duelo_vs_clasico.json "%PY%" scripts\comparar.py --a ppo:%L%\duelo\last.pt --players 2 --matches 2000 --workers 5 --json %C%\duelo_vs_clasico.json
if not exist %C%\duelo_vs_teoria_de_juegos.json "%PY%" scripts\comparar.py --a ppo:%L%\duelo\last.pt --b cfr:%L%\cfr\heads_up.npz --players 2 --matches 2000 --workers 5 --json %C%\duelo_vs_teoria_de_juegos.json
if not exist %C%\teoria_de_juegos_vs_clasico.json "%PY%" scripts\comparar.py --a cfr:%L%\cfr\heads_up.npz --players 2 --matches 2000 --workers 5 --json %C%\teoria_de_juegos_vs_clasico.json
if not exist %C%\teoria_de_juegos_largo_vs_corto.json "%PY%" scripts\comparar.py --a cfr:%L%\cfr\heads_up.npz --b cfr:runs\cfr\heads_up.npz --players 2 --matches 2000 --workers 5 --json %C%\teoria_de_juegos_largo_vs_corto.json
if not exist %F%\fuentes_vs_clasico.png "%PY%" scripts\figura_fuentes.py --json %C%\senas_vs_clasico.json %C%\sin_senas_vs_clasico.json %C%\espias_vs_clasico.json --nombres "IA con senas" "IA sin senas" "IA con espias" --titulo "De donde sacan los puntos las IA contra el jugador clasico" --out %F%\fuentes_vs_clasico.png
if not exist %F%\fuentes_mano_a_mano.png "%PY%" scripts\figura_fuentes.py --json %C%\duelo_vs_clasico.json %C%\teoria_de_juegos_vs_clasico.json --nombres "IA de mano a mano" "Bot de teoria de juegos" --titulo "Mano a mano contra el jugador clasico" --out %F%\fuentes_mano_a_mano.png
if exist runs\duelo\last.pt if not exist %C%\duelo_largo_vs_corto.json "%PY%" scripts\comparar.py --a ppo:%L%\duelo\last.pt --b ppo:runs\duelo\last.pt --players 2 --matches 2000 --workers 5 --json %C%\duelo_largo_vs_corto.json
title 9/9 Ligas y reporte
set EXTRA=
if exist runs\duelo\last.pt set EXTRA=ppo:runs\duelo\last.pt
if not exist %L%\liga\liga.json "%PY%" scripts\league.py --runs runs\parejas_senas runs\parejas_sin_senas runs\parejas_espias %L%\parejas_senas %L%\parejas_sin_senas %L%\parejas_espias --per-run 1 --matches 400 --out %L%\liga
if not exist %L%\liga_mano_a_mano\liga.json "%PY%" scripts\league.py --players 2 --specs ppo:%L%\duelo\last.pt %EXTRA% cfr:%L%\cfr\heads_up.npz cfr:runs\cfr\heads_up.npz --matches 400 --out %L%\liga_mano_a_mano
if not exist %L%\reporte\resumen.json "%PY%" scripts\report.py --runs %L% --out %L%\reporte
echo listo> %L%\listo.txt
title Todo terminado
pause
