# ForgeNPU

`ForgeNPU` e' il nome del progetto e del repository GitHub. Il package Python resta `create_npu`.

Bootstrap eseguibile del progetto descritto in `progetto_npu_ai.md`: prende un requisito in linguaggio naturale, lo converte in una specifica strutturata, genera piu' architetture candidate di NPU, produce seed RTL verificabili e salva score, artifact e diagnostica dell'ambiente.

## Stato attuale

Il progetto copre oggi un MVP esteso dei primi step della roadmap:

- parsing euristico del requisito;
- generazione di piu' candidate architecture (`balanced`, `throughput_max`, `efficiency`);
- scoring e selezione automatica del candidato migliore;
- generazione seed RTL di `mac_unit`, `processing_element`, `systolic_tile`, `dma_engine`, `scratchpad_controller`, `tile_compute_unit`, `scheduler` e `top_npu`;
- controllo seed del `scheduler` configurabile su numero di slot, iterazioni di load/compute e clear finale;
- `top_npu` parametrico su `TILE_COUNT` con wiring multi-tile seed e broadcast control-path verso piu' `tile_compute_unit`;
- testbench SystemVerilog per i moduli seed del compute cluster e del top-level;
- golden model Python con vettori di verifica salvati negli artifact;
- report di esecuzione con trace del `scheduler`, metriche del path memoria/compute e stima del throughput effettivo del `top_npu`;
- harness locale per lint, simulazione e sintesi con fallback esplicito;
- backend LLM opzionale con prompt/artifact preparatori e fallback controllato;
- comando `doctor` per diagnosticare tool EDA e stato backend LLM.

Il generatore RTL resta ancora seed-based: oggi produce un `top_npu` verificabile con `scheduler`, `dma_engine`, `scratchpad_controller` e `tile_compute_unit`, gia' predisposto a instanziare piu' tile identici, ma non implementa ancora uno scale-out completo con partizionamento reale dei dati e interconnect dedicato.

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

## Output della Run

Ogni run scrive una directory `runs/output_<timestamp>/` con:

- `result.json`: risultato finale con candidato selezionato, score e ambiente;
- `candidates.json`: archivio dei tentativi valutati con summary del report per candidato;
- `environment.json`: disponibilita' tool EDA e stato backend LLM;
- `candidates/<candidate_id>/design_intent.md`: intento progettuale del candidato;
- `candidates/<candidate_id>/rtl/`: seed RTL;
- `candidates/<candidate_id>/tb/`: testbench;
- `candidates/<candidate_id>/verification_vectors.json`: casi per il golden model Python;
- `candidates/<candidate_id>/execution_report.json`: trace degli stati del `scheduler`, metriche memoria/compute e stima di throughput del `top_npu`;
- `candidates/<candidate_id>/logs/`: log dei tool e del reference check;
- `candidates/<candidate_id>/llm_request.json`: prompt/artifact preparato quando si richiede `--generator-backend llm`.

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

## Backend LLM

Il backend LLM e' opzionale e si abilita via CLI:

```bash
python3 -m create_npu.cli run \
  --requirement "Voglio una NPU INT8 da 10 TOPS per dense GEMM." \
  --generator-backend llm \
  --llm-model gpt-test
```

Comportamento:

- se il package `openai` o `OPENAI_API_KEY` mancano, il sistema salva `llm_request.json` e ripiega sul backend euristico;
- se la configurazione LLM esiste ma `CREATE_NPU_ENABLE_LIVE_LLM` non vale `1`, il sistema non tenta chiamate live;
- il fallback e' riportato in `environment.json`, `result.json` e nelle note del bundle generato.

## Struttura

- `create_npu/`: package Python principale
- `examples/requirements/`: esempi di input utente
- `runs/`: artifact prodotti dalle run
- `tests/`: test unitari di base

## Prossimi passi consigliati

1. Abilitare `iverilog` per compilazione e simulazione reali.
2. Abilitare `verilator` e `yosys` per lint e sintesi automatici.
3. Migliorare `scratchpad_controller` con primitive di memoria piu' realistiche.
4. Modellare bandwidth e occupazione memoria nelle metriche.
5. Integrare chiamate LLM live per la sintesi di candidate RTL oltre il seed statico.
