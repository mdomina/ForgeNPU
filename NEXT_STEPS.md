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

## Priorita' Immediata

- [x] Estendere i report con stati del `scheduler` e metriche del path memoria/compute.
- [x] Introdurre un controllo meno rigido del `scheduler` rispetto alla sequenza fissa corrente.
- [x] Stimare throughput architetturale del sistema completo a livello `top_npu`.
- [x] Preparare il `top_npu` a scalare oltre un singolo tile seed.

## Verifica Hardware Reale

- [ ] Abilitare `iverilog` per compilazione e simulazione reali.
- [ ] Abilitare `verilator` per lint automatico.
- [ ] Abilitare `yosys` per sintesi logica.
- [ ] Salvare nei report metriche di compilazione, simulazione e sintesi per ogni candidato.

## Evoluzione del Compute Cluster

- [ ] Estendere `systolic_tile` oltre il caso `2x2`.
- [ ] Introdurre configurazione parametrica reale di tile size.
- [ ] Aggiungere gestione del flush della pipeline del tile.
- [ ] Separare chiaramente control-path e data-path del cluster.

## Memoria e Movimento Dati

- [ ] Migliorare `scratchpad_controller` con piu' banchi o doppio buffering.
- [ ] Introdurre primitive di load/store piu' vicine a un flusso NPU reale.
- [ ] Modellare bandwidth e occupazione memoria nelle metriche.

## Top-Level NPU

- [x] Implementare un primo `scheduler`.
- [x] Definire `top_npu` che assembli compute cluster, memoria e controllo.
- [x] Aggiungere test end-to-end del top-level.
- [x] Stimare throughput architetturale del sistema completo.

## Requirement System

- [ ] Estendere il parser con memoria disponibile, area target, latenza e bandwidth.
- [ ] Supportare piu' workload oltre `dense_gemm` e `transformer`.
- [ ] Ridurre le assunzioni implicite aumentando i campi strutturati del requirement.

## LLM Integration

- [ ] Attivare davvero il backend LLM live oltre il fallback euristico.
- [ ] Usare prompt strutturati per generare varianti RTL candidate.
- [ ] Confrontare output LLM con i seed RTL attuali tramite scoring automatico.

## Dataset e Learning Loop

- [ ] Salvare ogni run come sample riusabile per training/benchmark.
- [ ] Tenere traccia di candidati buoni e cattivi con score e log completi.
- [ ] Aggiungere best-of-N automatico.
- [ ] Preparare il progetto per rejection sampling o RL da feedback EDA.

## Operativita'

- [x] Aggiungere `Dockerfile` o ambiente container ripetibile.
- [x] Preparare esecuzione CI su server dedicato.
- [x] Aggiungere benchmark di regressione per evitare rotture del flow.

## Definizione di Done per il Prossimo Milestone

Il prossimo milestone puo' essere considerato chiuso quando:

- i report espongono almeno gli stati del `scheduler` e gli eventi principali del path memoria/compute;
- esiste una stima architetturale del throughput del `top_npu` e non solo del tile;
- il top-level e' pronto a scalare oltre il caso seed a singolo tile;
- i test del cluster e del top-level continuano a passare dopo l'estensione del controllo;
- la pipeline continua a selezionare automaticamente il candidato migliore;
- i report in `runs/` restano leggibili e confrontabili.
