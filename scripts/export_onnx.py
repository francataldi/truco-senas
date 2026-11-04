import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ppo.net import load_checkpoint
from agents.ppo.onnx_export import export_onnx, verify_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", type=int, default=2000)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    out = Path(args.out) if args.out else checkpoint.with_suffix(".onnx")
    net, rules, extra = load_checkpoint(str(checkpoint))
    export_onnx(net, out, rules=rules, opset=args.opset, extra={"checkpoint": checkpoint.as_posix(), **extra})
    size = out.stat().st_size / 1024
    print(f"exportado {out} ({size:.0f} KB, opset {args.opset}, entradas obs[batch, {net.obs_dim}] y mask[batch, {net.n_actions}])")
    if args.no_verify:
        return
    report = verify_onnx(net, out, rules, samples=args.samples, seed=args.seed, atol=args.atol, bench=args.bench)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        print("los logits de onnxruntime no coinciden con torch", file=sys.stderr)
        sys.exit(1)
    line = f"ok: {report['samples']} observaciones, diferencia máxima en logits {report['max_abs_logits']:.2e}, misma acción greedy en el 100%"
    if "latency_ms" in report:
        lat = report["latency_ms"]
        line += f", {lat['torch']:.3f} ms en torch vs {lat['onnxruntime']:.3f} ms en onnxruntime por decisión"
    print(line)


if __name__ == "__main__":
    main()
