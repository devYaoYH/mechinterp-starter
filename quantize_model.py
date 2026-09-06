"""Quantize a dense checkpoint to 4-bit, save it, and report what it cost.

    python quantize_model.py --model Qwen/Qwen2.5-1.5B-Instruct --out ./q/qwen-nf4

Read the fidelity report before running experiments on the result. Below ~0.95
top-1 agreement, treat it as a different model and reproduce headline results
on dense weights.
"""
import json
import os

from mechinterp import loading
from mechinterp import quantization as Q

PROBES = ["The capital of France is", "The capital of Japan is",
          "The chemical symbol for gold is", "7 times 6 equals", "A shark is a",
          "The largest planet in our solar system is",
          "Question: What is the capital of Peru?\nAnswer:",
          "In 1969, humans first landed on the"]


def main():
    args = loading.cli(out=dict(required=True, help="directory for the 4-bit weights"),
                       quant_type=dict(default="nf4", choices=["nf4", "fp4"]),
                       no_double_quant=dict(action="store_true"),
                       skip_fidelity=dict(action="store_true"))
    ref = None
    if not args.skip_fidelity:
        print(f"loading dense {args.model} to capture a reference...")
        dense, tok = loading.load_hf(args.model, "none")
        print(f"  dense footprint {dense.get_memory_footprint() / 1e9:.2f} GB")
        ref = Q.capture_reference(dense, tok, PROBES)
        Q.free(dense)              # must free before the quantized load at large sizes

    print(f"quantizing -> {args.out} ({args.quant_type})...")
    out_dir, footprint = Q.quantize_and_save(args.model, args.out, args.quant_type,
                                             double_quant=not args.no_double_quant)
    print(f"  saved, footprint {footprint / 1e9:.2f} GB")

    qmodel, qtok = loading.load_hf(out_dir, "prequantized")
    print(f"  reloaded from disk, footprint {qmodel.get_memory_footprint() / 1e9:.2f} GB")

    if ref is not None:
        rep = Q.compare_against(qmodel, qtok, PROBES, ref)
        print("\n" + Q.format_report(rep, f"{args.quant_type} vs dense"))
        rep["layer_cos"] = [float(x) for x in rep["layer_cos"]]
        with open(os.path.join(out_dir, "fidelity_report.json"), "w") as f:
            json.dump(rep, f, indent=1)

    print(f"\nUse it with:  loading.load_hf({out_dir!r}, 'prequantized')")


if __name__ == "__main__":
    main()
