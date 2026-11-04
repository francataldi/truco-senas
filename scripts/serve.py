import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from demo.api import Settings, create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs", action="append", default=None)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--partner", default=None)
    parser.add_argument("--rivals", default=None)
    parser.add_argument("--team-a", default=None)
    parser.add_argument("--team-b", default=None)
    parser.add_argument("--dictionary-hands", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    first = f"ppo:{Path(args.checkpoint[0]).as_posix()}" if args.checkpoint else None
    settings = Settings(
        roots=[Path(r) for r in (args.runs or ["runs"])],
        allowed=[Path(p) for p in args.checkpoint],
        partner=args.partner or first,
        rivals=args.rivals,
        team_a=args.team_a or first,
        team_b=args.team_b,
        dictionary_hands=args.dictionary_hands,
        seed=args.seed,
    )
    app = create_app(settings)
    catalog = app.state.catalog
    agents = catalog.discover()
    print(f"agentes disponibles ({len(agents)}):")
    for info in agents:
        print(f"  {info.spec:48s} {info.label}")
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"\ntruco con señas en {url}  (ctrl+c para cortar)", flush=True)
    if args.open:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
