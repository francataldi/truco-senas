import pytest

from agents.behavior import SOURCES, compare
from truco.rules import Rules


@pytest.mark.parametrize("players", [4, 2])
def test_points_by_source_add_up(players):
    rules = Rules(num_players=players, signals=players == 4)
    result = compare(rules, "heuristic", "random", 6, seed=3)
    a, b = result["equipo_a"], result["equipo_b"]
    for side in (a, b):
        assert sum(side["puntos_por_mano_por_fuente"].values()) == pytest.approx(side["puntos_por_mano"], abs=1e-3)
    total = sum(result["diferencia_por_fuente"][s]["diff"] for s in SOURCES)
    assert total == pytest.approx(result["diferencia_por_mano"]["diff"], abs=1e-3)
    assert result["diferencia_por_mano"]["diff"] == pytest.approx(a["puntos_por_mano"] - b["puntos_por_mano"], abs=1e-3)
    assert result["partidos"] == 6 and 0 <= result["gana_a"] <= 1
    for side in (a, b):
        for group in ("envido", "truco"):
            for key, value in side[group].items():
                if value is not None and key != "envido_medio_al_cantar":
                    assert 0 <= value <= 1


def test_duplicate_mirror_is_balanced():
    result = compare(Rules(), "heuristic", "heuristic", 40, seed=1)
    lo, hi = result["gana_a_ci95"]
    assert lo < 0.5 < hi
