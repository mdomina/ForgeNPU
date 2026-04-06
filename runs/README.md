Le run della pipeline scrivono qui gli artifact (`result.json`, RTL seed, testbench, `compiled_program.json` con shape/operator plan, `verification_vectors.json` con stress top-level e interni, `coverage_report.json`, log dei tool).
La root puo' contenere anche `dataset/` con archivi JSONL, contatori cumulativi, reward EDA e label accept/reject per il learning loop.
