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
- [ ] Estendere il compiler a shape/operatori reali oltre i descrittori seed sintetici correnti.
- [ ] Collegare il compiler a benchmark workload-specifici piu' ricchi.

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
- i test Python, il benchmark locale e la regressione Docker full-toolchain restano verdi.
