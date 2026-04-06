# ForgeNPU Future Steps

`ForgeNPU` e' il nome del progetto e del repository GitHub.

Nota:

- il repository GitHub e' `ForgeNPU`
- il package Python resta `create_npu`
- la CLI attuale non cambia

## Obiettivo del File

Questa e' la checklist pratica dei prossimi step da seguire, in ordine consigliato.

## Milestone Appena Chiusa

- [x] Implementare un `dma_engine` minimo che scriva vettori nello `scratchpad_controller`.
- [x] Collegare `dma_engine -> scratchpad_controller -> tile_compute_unit`.
- [x] Aggiungere golden model Python e testbench del path DMA.
- [x] Rendere il `tile_compute_unit` il top del piccolo compute cluster con ingressi piu' realistici.
- [x] Richiudere il loop del cluster con test automatici e run pipeline end-to-end.
- [x] Implementare un primo `scheduler` con sequenza `DMA -> load -> compute -> clear`.
- [x] Collegare `scheduler` e `tile_compute_unit` in un primo `top_npu`.
- [x] Aggiungere test end-to-end del top-level con una sequenza minima di inferenza.
- [x] Generalizzare le primitive di memoria con descrittori minimi di `load/store`, stride e burst count.
- [x] Rendere osservabile il writeback burst del `top_npu` con payload segmentato e `valid_mask`.
- [x] Richiudere il flow completo nel container Docker con `verilator`, `iverilog` e `yosys` verdi.
- [x] Aggiungere uno stato esplicito di `FLUSH` nel `scheduler` e propagarlo fino al tile.
- [x] Verificare `systolic_tile` anche oltre il caso `2x2` con un testbench rettangolare dedicato.
- [x] Derivare la shape di default del seed RTL dalla tile architetturale del candidato, evitando il `2x2` fisso nel cluster.

## Priorita' Immediata

- [x] Estendere i report con stati del `scheduler` e metriche del path memoria/compute.
- [x] Introdurre un controllo meno rigido del `scheduler` rispetto alla sequenza fissa corrente.
- [x] Stimare throughput architetturale del sistema completo a livello `top_npu`.
- [x] Preparare il `top_npu` a scalare oltre un singolo tile seed.

## Verifica Hardware Reale

- [x] Abilitare `iverilog` per compilazione e simulazione reali.
- [x] Abilitare `verilator` per lint automatico.
- [x] Abilitare `yosys` per sintesi logica.
- [x] Salvare nei report metriche di compilazione, simulazione e sintesi per ogni candidato.

## Evoluzione del Compute Cluster

- [x] Estendere `systolic_tile` oltre il caso `2x2`.
- [x] Introdurre configurazione parametrica reale di tile size senza riduzione seed per la verifica locale.
- [x] Aggiungere gestione del flush della pipeline del tile.
- [x] Separare chiaramente control-path e data-path del cluster.

## Memoria e Movimento Dati

- [x] Migliorare `scratchpad_controller` con piu' banchi o doppio buffering.
- [x] Collegare la selezione dei banchi allo `scheduler` e al `top_npu`.
- [x] Introdurre una prima primitiva `STORE` top-level con writeback osservabile e metriche dedicate.
- [x] Generalizzare le primitive di load/store con descrittori, burst e destinazioni risultato piu' realistiche.
- [x] Modellare bandwidth e occupazione memoria nelle metriche.

## Top-Level NPU

- [x] Implementare un primo `scheduler`.
- [x] Definire `top_npu` che assembli compute cluster, memoria e controllo.
- [x] Aggiungere test end-to-end del top-level.
- [x] Stimare throughput architetturale del sistema completo.
- [x] Propagare `TILE_COUNT` reale nel seed RTL senza l'attuale riduzione locale.
- [x] Modellare un interconnect seed piu' dedicato per il traffico multi-tile e il broadcast del controllo.
- [x] Introdurre handshake e backpressure nell'interconnect seed per DMA/load/store multi-tile.

## Requirement System

- [x] Estendere il parser con memoria disponibile, area target, latenza e bandwidth.
- [x] Supportare piu' workload oltre `dense_gemm` e `transformer`.
- [x] Ridurre le assunzioni implicite aumentando i campi strutturati del requirement.

## LLM Integration

- [x] Attivare davvero il backend LLM live oltre il fallback euristico.
- [x] Usare prompt strutturati per generare varianti RTL candidate.
- [x] Confrontare output LLM con i seed RTL attuali tramite scoring automatico.

## Compiler e Mapping

- [x] Introdurre un layer compiler/mapping che traduca workload e requirement in un programma seed `LOAD/COMPUTE/STORE`.
- [x] Estendere il compiler a shape/operatori reali oltre i descrittori seed sintetici correnti.
- [x] Collegare il compiler a benchmark workload-specifici piu' ricchi.

## Backlog Ispirato a Gemmini https://github.com/ucb-bar/gemmini.git

Nota:

- usare Gemmini come baseline e reference architecture, non come dipendenza diretta del flow corrente;
- evitare per ora integrazione con `Chipyard`, `RoCC` o toolchain `Chisel`, privilegiando concetti riassorbibili nei seed RTL SystemVerilog, nel compiler e nel reporting gia' presenti.

- [x] Introdurre un `accumulator_buffer` esplicito separato dallo `scratchpad_controller`, con supporto a partial sums persistenti, cast/scaling in readback e hook minimi per bias.
- [x] Estendere `tile_compute_unit` e `systolic_tile` con una selezione reale del dataflow `weight_stationary` / `output_stationary`, propagata dal `compiled_program` fino ai report.
- [ ] Aggiungere un path minimo di `preload` e `transpose` per studiare casi `output_stationary` senza forzare tutto il mapping sul solo seed systolic attuale.
- [ ] Evolvere lo `scheduler` verso code decoupled `load/store/execute` con hazard tracking minimale e metriche di overlap osservabili tra accesso memoria e compute.
- [ ] Introdurre nel compiler primitive di loop tiled per `gemm` e `convolution`, con doppio buffering esplicito e stima dell'occupancy del cluster durante l'esecuzione.
- [ ] Aggiungere benchmark e casi di riferimento "Gemmini-like" per confrontare shape, dataflow, memoria locale e throughput stimato del candidato contro una baseline esterna nota.
- [ ] Salvare nei report un delta esplicito tra requirement, mapping scelto e reference architecture family, cosi' da rendere visibile quando un candidato converge o diverge da una classe tipo Gemmini.

## Backlog Ispirato a tiny-NPU https://github.com/harishsg993010/tiny-NPU.git

Nota:

- usare `tiny-NPU` come reference implementation vicina allo stack attuale (`SystemVerilog` + compiler Python + golden model + Verilator), non come top-level da assorbire integralmente;
- privilegiare i blocchi gia' maturi e riusabili come pattern (`compiler`, `memory planner`, `scoreboard`, `gemm_ctrl`, wrapper di simulazione), evitando di copiare nel breve i path ancora parzialmente stub del top-level originale.

- [ ] Introdurre un formato di `tensor_descriptor` piu' ricco nel compiler ForgeNPU, con shape, dtype, size, base address e metadati sufficienti per una futura esecuzione `graph_mode` o `operator_mode` piu' autonoma.
- [ ] Aggiungere un `memory planner` con allocazione SRAM, `free` e riuso basato su liveness dei tensori/operatori, riportando nei report peak SRAM con e senza reuse.
- [ ] Evolvere il compiler da solo `LOAD/COMPUTE/STORE` seed a lowering per operatori concreti, partendo da `Gemm`, `Conv` via `im2col`, `Relu`, `Reduce`, `Concat`, `Slice` e `Pool`.
- [ ] Introdurre una modalita' di dispatch con `scoreboard` e `barrier` espliciti, separando meglio issue, dipendenze e completamento dei motori rispetto allo scheduler seed corrente.
- [ ] Aggiungere performance counters piu' granulari per `compute`, `dma`, `stall`, `overlap` e occupazione effettiva del cluster, cosi' da rendere confrontabili i candidati anche oltre pass/fail e throughput grezzo.
- [ ] Introdurre un path di accumulo dedicato stile `ACC SRAM` o `accumulator_buffer` per il `gemm_ctrl`, cosi' da supportare K-tiling reale, partial sums persistenti e writeback differito.
- [ ] Preparare un wrapper di simulazione dedicato per un futuro `graph_mode` o `operator_mode`, separato dal `top_npu` principale, per verificare compiler, descriptors, DMA e motori senza bloccare l'integrazione top-level definitiva.
- [ ] Rafforzare la regressione con casi end-to-end compiler -> golden -> RTL su grafi piccoli deterministici e fuzz workload-aware, includendo mismatch di shape, overflow SRAM, reuse aggressivo e overlap `DMA+compute`.

## Verifica e Stress

- [x] Rafforzare la verifica con casi randomizzati e stress test su backpressure, flush e multi-tile.
- [x] Estendere i casi stress ai moduli interni oltre `top_npu`.
- [x] Integrare coverage o check piu' quantitativi oltre l'attuale golden model.

## Dataset e Learning Loop

- [x] Salvare ogni run come sample riusabile per training/benchmark.
- [x] Tenere traccia di candidati buoni e cattivi con score e log completi.
- [x] Aggiungere best-of-N automatico.
- [x] Preparare il progetto per rejection sampling o RL da feedback EDA.

## Operativita'

- [x] Aggiungere `Dockerfile` o ambiente container ripetibile.
- [x] Preparare esecuzione CI su server dedicato.
- [x] Aggiungere benchmark di regressione per evitare rotture del flow.

## Definizione di Done per il Prossimo Milestone

Il prossimo milestone puo' essere considerato chiuso quando:

- il flow compila un programma seed `LOAD/COMPUTE/STORE` coerente con workload, batch e mapping base del candidato;
- il programma compilato e' persistito negli artifact della run e riportato in `design_intent` e `execution_report`;
- i vettori di verifica top-level riflettono il programma compilato senza rompere il flow seed RTL corrente;
- la verifica include casi stress deterministici per backpressure, flush e multi-tile senza alterare le metriche smoke del benchmark;
- i test Python, il benchmark locale e la regressione Docker full-toolchain restano verdi;
- il flow espone anche un report di coverage quantitativa sui casi di riferimento oltre al solo golden model.
