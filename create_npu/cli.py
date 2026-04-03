import argparse
import json
from pathlib import Path

from create_npu.benchmark import run_regression_benchmark
from create_npu.environment import collect_environment_snapshot
from create_npu.pipeline import CreateNPUPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CLI for createNPU.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Parse, plan and execute one MVP run.")
    run_parser.add_argument(
        "--requirement",
        required=True,
        help="Natural-language requirement for the target accelerator.",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: runs/output_<timestamp>",
    )
    run_parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Numero di candidati architetturali da generare e valutare.",
    )
    run_parser.add_argument(
        "--generator-backend",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Backend di generazione richiesto.",
    )
    run_parser.add_argument(
        "--llm-model",
        default=None,
        help="Modello da usare per il backend LLM opzionale.",
    )

    plan_parser = subparsers.add_parser(
        "plan", help="Parse and plan architecture, then print the result."
    )
    plan_parser.add_argument(
        "--requirement",
        required=True,
        help="Natural-language requirement for the target accelerator.",
    )
    plan_parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Numero di candidati da pianificare.",
    )
    plan_parser.add_argument(
        "--generator-backend",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Backend di generazione richiesto.",
    )
    plan_parser.add_argument(
        "--llm-model",
        default=None,
        help="Modello da usare per il backend LLM opzionale.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="Mostra disponibilita' tool EDA e stato backend LLM."
    )
    doctor_parser.add_argument(
        "--generator-backend",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Backend richiesto per la diagnosi.",
    )
    doctor_parser.add_argument(
        "--llm-model",
        default=None,
        help="Modello da usare per la diagnosi LLM.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Esegue benchmark di regressione del flow end-to-end."
    )
    benchmark_parser.add_argument(
        "--output-dir",
        default="runs/output_regression_benchmark",
        help="Directory di output del benchmark di regressione.",
    )
    benchmark_parser.add_argument(
        "--llm-model",
        default="gpt-test",
        help="Modello da usare per il caso benchmark di fallback LLM.",
    )
    benchmark_parser.add_argument(
        "--require-full-toolchain",
        action="store_true",
        help="Fallisce se Verilator, Icarus e Yosys non sono disponibili o non passano.",
    )

    args = parser.parse_args()

    if args.command == "run":
        pipeline = CreateNPUPipeline()
        output_dir = Path(args.output_dir) if args.output_dir else None
        result = pipeline.run(
            requirement_text=args.requirement,
            output_dir=output_dir,
            num_candidates=args.num_candidates,
            generator_backend=args.generator_backend,
            llm_model=args.llm_model,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return

    if args.command == "plan":
        pipeline = CreateNPUPipeline()
        result = pipeline.run(
            requirement_text=args.requirement,
            output_dir=Path("runs/plan_preview"),
            num_candidates=args.num_candidates,
            generator_backend=args.generator_backend,
            llm_model=args.llm_model,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return

    if args.command == "doctor":
        print(
            json.dumps(
                collect_environment_snapshot(
                    requested_backend=args.generator_backend,
                    llm_model=args.llm_model,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "benchmark":
        payload = run_regression_benchmark(
            output_dir=Path(args.output_dir),
            require_full_toolchain=bool(args.require_full_toolchain),
            llm_model=args.llm_model,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload["passed"]:
            raise SystemExit(1)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
