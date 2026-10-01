# What the Kernel Sees

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23074717.svg)](https://doi.org/10.5281/zenodo.23074717)

Data and code for the preprint **"What the Kernel Sees: An Exploratory Study of Quantized LLM Inference on a
6 GB Consumer Laptop, and Pitfalls in Comparing Local Runtimes"** by Chuka Josemaria Uzo
([ORCID 0009-0009-9411-2858](https://orcid.org/0009-0009-9411-2858)).

The archived version of record (paper and data) is on Zenodo: https://doi.org/10.5281/zenodo.23074717

Measurements were collected on 28 February 2024 for the author's MSc thesis
(https://doi.org/10.5281/zenodo.13829349) and re-analysed for the paper. No new experiments were run.

## Contents

| Path | Contents |
| --- | --- |
| `paper/` | The paper (PDF) |
| `per_run_counters_and_phases.csv` | One row per run (45 runs): all `perf stat` counters, wall/user/system time, and prompt-processing and generation tokens and times where llama.cpp logged them |
| `gpu_telemetry_wasmedge.csv` | Per-second GPU and memory samples for every WasmEdge run, with corrected column names |
| `raw_logs/PYTHON RESULTS/` | Original `perf stat` output and full run log (llama.cpp messages, timings, model answer) for each Python / llama-cpp-python run |
| `raw_logs/RUST-WASM RESULTS/` | The same for each WasmEdge run, plus the original `gpu_stat (n).csv` files |
| `make_figs.py` | Regenerates the paper's figures (run from this folder; needs numpy, scipy, matplotlib) |

## Notes

- The original `gpu_stat` headers are mislabelled: the column headed "% Used GPU Memory" holds GPU memory
  used in MiB; "% Used RAM" holds RAM used in GB; "Cached RAM" holds buffer/cache in GB; and
  "Buffered RAM" holds available memory in GB. `gpu_telemetry_wasmedge.csv` uses the correct names.
  Timestamps have minute resolution; samples were taken about once per second.
- WasmEdge's detailed inference logging was switched off for the Mixtral, Llama-2 and Mistral runs,
  so phase timings exist only for the WasmEdge Phi-2 and Zephyr runs.
- The input article (LaRocco, CNBC, 31 January 2024) is copyrighted. Every passage of ten or more
  consecutive words copied from it, in prompts or echoed in model outputs, has been replaced with
  `[ARTICLE TEXT REMOVED]`.
- WasmEdge logs contain terminal control sequences from the `watch` command used for GPU logging.
- The Phi-2 WasmEdge files are numbered 1, 3, 5, 6, 7; these are the five runs reported, in order.
- Zephyr produced no output under Python, so that folder is empty.

## Citation

> Uzo, C. J. (2026). *What the Kernel Sees: An Exploratory Study of Quantized LLM Inference on a 6 GB
> Consumer Laptop, and Pitfalls in Comparing Local Runtimes.* Preprint, Zenodo.
> https://doi.org/10.5281/zenodo.23074717

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE`.
