# Design Intent

## Requirement

Voglio una NPU INT8 da 1000 TFLOPS ottimizzata per inferenza transformer, con consumo massimo di 250 W e supporto batch 1-16.

## Parsed Spec

- Precisione: INT8
- Throughput target: 1000.0 TFLOPS
- Potenza: 250.0 W
- Workload: transformer
- Batch: 1-16
- Interfacce: AXI4, DMA, scratchpad_sram
- Tecnologia: generic_5nm
- Frequenza: 1000.0 MHz

## Assunzioni

- Interfacce non dichiarate: assumo AXI4, DMA e scratchpad SRAM.
- Nodo tecnologico non specificato: uso una baseline generica 5nm ASIC.
- Frequenza target non specificata: assumo 1000 MHz per la stima iniziale.
- Interpreto TFLOPS su INT8 come throughput di operazioni equivalente a TOPS.

## Ambiguita'

- Nodo tecnologico assente.
- Frequenza target assente.

## Candidate Architecture

- Famiglia: tiled_systolic_transformer
- Tile: 64x64
- Tile count: 132
- Mesh logica: 768x704
- PE count: 540672
- SRAM per tile: 768 KB
- Global buffer: 33 MB
- Bus width: 2048 bit
- Throughput stimato: 1081.34 TOPS

## Rationale

- Candidato valutato: balanced.
- Famiglia architetturale scelta: tiled_systolic_transformer.
- Numero di MAC stimato con formula throughput / (2 * frequenza), pari a circa 500000 unita'.
- Dimensionamento iniziale in mesh 768x704 con tile 64x64.
- Stimato throughput teorico di 1081.34 TOPS a 1000 MHz.
- Potenza stimata iniziale: 376.4 W.
- Area stimata iniziale: 89.4 mm2.
- Batch massimo 16: aumento buffer locale e ampiezza del fabric.
- Budget di potenza rilevato: 250 W.
- Profilo balanced: compromesso iniziale tra throughput, banda e consumi.

## Planned Modules

- mac_unit
- processing_element
- systolic_tile
- scratchpad_controller
- dma_engine
- scheduler
- top_npu
