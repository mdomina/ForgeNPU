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
- Tile count: 110
- Mesh logica: 704x640
- PE count: 450560
- SRAM per tile: 896 KB
- Global buffer: 27 MB
- Bus width: 2560 bit
- Throughput stimato: 1036.29 TOPS

## Rationale

- Candidato valutato: throughput_max.
- Famiglia architetturale scelta: tiled_systolic_transformer.
- Numero di MAC stimato con formula throughput / (2 * frequenza), pari a circa 434783 unita'.
- Dimensionamento iniziale in mesh 704x640 con tile 64x64.
- Stimato throughput teorico di 1036.29 TOPS a 1150 MHz.
- Potenza stimata iniziale: 431.2 W.
- Area stimata iniziale: 84.4 mm2.
- Batch massimo 16: aumento buffer locale e ampiezza del fabric.
- Budget di potenza rilevato: 250 W.
- Profilo throughput_max: privilegio frequenza e banda rispetto al costo energetico.

## Planned Modules

- mac_unit
- processing_element
- systolic_tile
- scratchpad_controller
- dma_engine
- scheduler
- top_npu
- prefetch_buffer
