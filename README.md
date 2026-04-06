# ForgeNPU

`ForgeNPU` e' il nome del progetto e del repository GitHub. Il package Python resta `create_npu`.

Bootstrap eseguibile del progetto descritto in `progetto_npu_ai.md`: prende un requisito in linguaggio naturale, lo converte in una specifica strutturata, genera piu' architetture candidate di NPU, produce seed RTL verificabili e salva score, artifact e diagnostica dell'ambiente.

## Stato attuale

Il progetto copre oggi un MVP esteso dei primi step della roadmap:

- parsing euristico del requisito con memoria disponibile, bandwidth, area, latenza e frequenza target;
- riconoscimento workload anche per `convolution` e `sparse_linear_algebra`, con alias pratici (`cnn`, `conv2d`, `spmm`, `sparse matmul`) oltre `dense_gemm` e `transformer`;
- specifica strutturata piu' ricca con `execution_mode`, `optimization_priority`, `offchip_memory_type`, `preferred_dataflow`, `sparsity_support`, `sequence_length` e `kernel_size`;
- generazione di piu' candidate architecture (`balanced`, `throughput_max`, `efficiency`);
- scoring e selezione automatica del candidato migliore;
- generazione seed RTL di `mac_unit`, `processing_element`, `systolic_tile`, `dma_engine`, `scratchpad_controller`, `tile_compute_unit`, `scheduler`, `cluster_control` e `top_npu`;
- `scratchpad_controller` seed con tracciamento di validita' e due banchi selezionabili, coerente con il timing del path DMA/load;
- `tile_compute_unit` seed capace di leggere e scrivere banchi distinti dello scratchpad per esperimenti di prefetch/ping-pong locali;
- `top_npu` seed che decodifica gli slot del programma in `bank + local_addr`, usando davvero il banking del tile nel flow end-to-end;
- `systolic_tile` seed verificato anche oltre il solo caso `2x2`, con testbench rettangolare dedicato per la parametrizzazione del tile;
- default `ROWS`/`COLS` del seed RTL propagati direttamente dalla tile architetturale del candidato;
- controllo seed del `scheduler` configurabile su numero di slot, iterazioni di load/compute, descrittori minimi di memoria e clear finale;
- primitive seed di `LOAD/STORE` con base address, stride e burst count, propagate nel `scheduler` e nel `top_npu`;
- stato esplicito `FLUSH` nel `scheduler`, propagato fino al `tile_compute_unit` e al `systolic_tile` per separare il drain/reset della pipeline dall'azzeramento degli accumulatori;
- `cluster_control` seed dedicato al routing del control-path tra `scheduler` e tile, con decode di bank/local address e broadcast per-tile delle primitive di DMA/load/compute/store/flush/clear;
- `cluster_interconnect` seed dedicato al fanout del data/control routing multi-tile, con handshake `valid/ready` sui path DMA/load/store, backpressure osservabile e aggregazione dei burst `STORE` del cluster;
- writeback `STORE` top-level segmentato per burst, con payload risultati e `valid_mask` osservabili a livello top-level;
- `top_npu` parametrico su `TILE_COUNT` che orchestra `scheduler`, `cluster_control`, `cluster_interconnect` e piu' `tile_compute_unit` senza incorporare direttamente il fanout del cluster;
- testbench SystemVerilog per i moduli seed del compute cluster e del top-level;
- golden model Python con vettori di verifica salvati negli artifact;
- report di esecuzione con trace del `scheduler`, metriche del path memoria/compute, occupancy dello scratchpad, traffico DMA/store, cicli di flush, bandwidth effettiva/teorica e stima del throughput effettivo del `top_npu`;
- report di esecuzione con profilo workload esplicito, requirement profile strutturato, famiglia architetturale preferita/selezionata e assunzioni persistite del parser;
- archivio dataset locale che salva ogni run come sample riusabile e traccia candidati `good`/`bad` con score, report e log dei tool;
- best-of-N automatico sopra i tre profili seed, con espansione deterministica di varianti quando `--num-candidates` supera i profili base;
- segnali di learning feedback derivati dall'EDA (`accept/reject`, reward, bucket di feedback e motivi di rejection) pronti per rejection sampling o RL offline;
- harness locale per lint, simulazione e sintesi con fallback esplicito, inclusa sintesi `yosys` bounded sui parametri dei casi `top_npu` per mantenere trattabile la regressione;
- verifica Python rinforzata con casi `top_npu_stress` randomizzati in modo deterministico per backpressure, flush e multi-tile, piu' stress dedicati per `scheduler`, `cluster_control` e `cluster_interconnect`;
- backend LLM opzionale con chiamata live, output JSON strutturato, override RTL mirati e fallback controllato;
- compiler seed che traduce workload e requirement in un programma `LOAD/COMPUTE/STORE` esplicito con mapping, shape reali, operator plan e descrittori persistiti;
- benchmark di regressione arricchito con casi workload-specifici per `transformer`, `convolution`, `sparse_linear_algebra` e fallback LLM, con check espliciti sul `compiled_program`;
- comando `doctor` per diagnosticare tool EDA e stato backend LLM.

Il generatore RTL resta seed-based come baseline, ma puo' ora affiancare una variante live LLM con override controllati di `processing_element.sv` e `systolic_tile.sv`, confrontandola automaticamente contro il seed euristico tramite lo scoring corrente. Sopra il seed hardware c'e' ora anche un primo layer di `compiler/mapping` che compila un programma `LOAD/COMPUTE/STORE` dal workload e lo persiste negli artifact e nei report, includendo shape reali per GEMM, convolution, transformer attention e sparse matmul insieme al mapping plan corrispondente. Il benchmark di regressione usa adesso anche casi workload-specifici per verificare che questi `compiled_program` siano coerenti su `transformer`, `convolution`, `sparse_linear_algebra` e sul path di fallback LLM. La verifica stress non si ferma piu' al solo `top_npu`: i manifest includono anche casi interni per `scheduler`, `cluster_control` e `cluster_interconnect`, con riepilogo dedicato nel report. Il cluster separa in modo esplicito control-path, interconnect e data-path del top-level, con banking, descrittori minimi, burst di writeback, flush della pipeline e backpressure multi-tile instradati dai moduli dedicati; inoltre la shape di default del cluster e il `TILE_COUNT` vengono propagati direttamente dalla tile architetturale reale, senza riduzioni intermedie. Le run vengono anche archiviate come dataset locale in `runs/dataset/` o nella root del benchmark corrente, con search summary best-of-N e learning feedback EDA pronti per loop di training offline. Il passo successivo con piu' impatto e' aggiungere coverage piu' quantitativa sopra il golden model.

## Quick Start

Vedi anche [SETUP.md](SETUP.md) per preparare una macchina target.
Vedi anche [NEXT_STEPS.md](NEXT_STEPS.md) per la TODO list dei prossimi milestone.

Clona il repository:

```bash
git clone https://github.com/mdomina/ForgeNPU.git
cd ForgeNPU
```

Esegui i test:

```bash
python3 -m unittest discover -s tests
```

Avvia l'ambiente Ubuntu Docker per sviluppo:

```bash
docker compose build dev
docker compose run --rm dev
```

Dentro il container puoi usare gli stessi comandi del progetto:

```bash
python3 -m unittest discover -s tests
python3 -m create_npu.cli doctor
python3 -m create_npu.cli benchmark --require-full-toolchain
```

La repository include anche una CI GitHub Actions in `.github/workflows/ci.yml` che:

- builda il container `dev`;
- esegue test unitari e `doctor` dentro Docker;
- lancia un benchmark di regressione end-to-end;
- carica gli artifact della run CI.

Lancia una run multi-candidato:

```bash
python3 -m create_npu.cli run \
  --requirement "Voglio una NPU INT8 da 1000 TFLOPS ottimizzata per inferenza transformer, con consumo massimo di 250 W e supporto batch 1-16." \
  --num-candidates 3
```

Diagnostica l'ambiente:

```bash
python3 -m create_npu.cli doctor --generator-backend llm --llm-model gpt-test
```

Esegui il benchmark di regressione:

```bash
python3 -m create_npu.cli benchmark \
  --output-dir runs/output_regression_benchmark \
  --require-full-toolchain
```

## Output della Run

Ogni run scrive una directory `runs/output_<timestamp>/` con:

- `result.json`: risultato finale con candidato selezionato, score e ambiente;
- `candidates.json`: archivio dei tentativi valutati con summary del report per candidato;
- `environment.json`: disponibilita' tool EDA e stato backend LLM;
- `candidates/<candidate_id>/design_intent.md`: intento progettuale del candidato;
- `candidates/<candidate_id>/rtl/`: seed RTL;
- `candidates/<candidate_id>/tb/`: testbench;
- `candidates/<candidate_id>/compiled_program.json`: programma seed compilato dal layer di mapping con shape workload, operator plan, slot, iterazioni, burst e descrittori;
- `candidates/<candidate_id>/verification_vectors.json`: casi per il golden model Python, inclusi stress case deterministici per `top_npu` e moduli interni;
- `candidates/<candidate_id>/execution_report.json`: trace degli stati del `scheduler`, metriche memoria/compute e stima di throughput del `top_npu`;
- `candidates/<candidate_id>/logs/`: log dei tool e del reference check;
- `candidates/<candidate_id>/llm_request.json`: richiesta completa al backend LLM con prompt, schema e moduli seed;
- `candidates/<candidate_id>/llm_response.json`: risposta raw serializzata del backend LLM;
- `candidates/<candidate_id>/llm_structured_output.json`: payload JSON parsato e validato dal flow;
- `candidates/<candidate_id>/backend_comparison.json`: confronto automatico tra seed euristico e variante LLM con delta di score e risultato selezionato;
- `candidates/<candidate_id>/heuristic_seed/` e `candidates/<candidate_id>/llm_variant/`: artifact separati quando il backend live LLM e' realmente attivo;
- `run_dataset_sample.json`: sample riusabile della run con requirement, candidato selezionato e summary del report;
- `candidate_dataset_samples.json`: sample dei candidati della run con label `good`/`bad`, score, report, path ai log e learning feedback EDA;

Nella root `runs/dataset/` vengono anche mantenuti:

- `run_samples.jsonl`: archivio append-only dei sample di run;
- `candidate_samples.jsonl`: archivio append-only dei sample di candidato;
- `manifest.json`: contatori aggregati di run, candidati buoni/cattivi, accepted/rejected e reward medio.

## Tool EDA supportati

Il progetto rileva automaticamente:

- `iverilog` per compilazione/simulazione;
- `verilator` per lint;
- `yosys` per sintesi.

Se non sono installati, la pipeline non fallisce: marca gli step come non disponibili e continua con score provvisorio. I suggerimenti correnti di installazione sono:

```bash
brew install icarus-verilog
brew install verilator
brew install yosys
```

Per un ambiente Linux ripetibile e' disponibile anche il `Dockerfile` Ubuntu del repository, gia' configurato con `python3`, `iverilog`, `verilator` e `yosys`.

## Backend LLM

Il backend LLM e' opzionale e si abilita via CLI:

```bash
python3 -m pip install -e ".[llm]"
export OPENAI_API_KEY=...
export CREATE_NPU_ENABLE_LIVE_LLM=1
python3 -m create_npu.cli run \
  --requirement "Voglio una NPU INT8 da 10 TOPS per dense GEMM." \
  --generator-backend llm \
  --llm-model gpt-test
```

Comportamento:

- il flow salva sempre `llm_request.json`; se la chiamata live riesce salva anche `llm_response.json` e `llm_structured_output.json`;
- il modello riceve prompt strutturati con spec, architettura e seed RTL dei moduli consentiti, e deve restituire JSON conforme allo schema atteso;
- gli override vengono applicati solo a `processing_element.sv` e `systolic_tile.sv`, mantenendo il resto del bundle seed invariato;
- quando il backend live e' attivo, la pipeline valuta sia il seed euristico sia la variante LLM e salva il confronto in `backend_comparison.json`;
- se `openai`, `OPENAI_API_KEY` o `CREATE_NPU_ENABLE_LIVE_LLM=1` mancano, oppure se la risposta non e' valida, il sistema ripiega automaticamente sul backend euristico;
- il fallback e la selezione finale sono riportati in `environment.json`, `result.json` e nelle note del bundle generato.

## Struttura

- `create_npu/`: package Python principale
- `examples/requirements/`: esempi di input utente
- `runs/`: artifact prodotti dalle run
- `tests/`: test unitari di base

## Prossimi passi consigliati

1. Evolvere lo scoring verso confronto multi-obiettivo e frontiera di Pareto su throughput, potenza e area.
2. Integrare coverage e check piu' quantitativi oltre il golden model attuale.
3. Espandere il dataset per includere metriche aggregate di coverage e failure modes, non solo score/run artifact.
