# Design Intent

## Requirement

Voglio una NPU INT8 da 50 TOPS con supporto transformer e batch 1-4.

## Parsed Spec

- Precisione: INT8
- Throughput target: 50.0 TOPS
- Potenza: None W
- Workload: transformer
- Batch: 1-4
- Interfacce: AXI4, DMA, scratchpad_sram
- Tecnologia: generic_5nm
- Frequenza: 1000.0 MHz

## Assunzioni

- Interfacce non dichiarate: assumo AXI4, DMA e scratchpad SRAM.
- Nodo tecnologico non specificato: uso una baseline generica 5nm ASIC.
- Frequenza target non specificata: assumo 1000 MHz per la stima iniziale.

## Ambiguita'

- Budget di potenza assente o non parsabile.
- Nodo tecnologico assente.
- Frequenza target assente.

## Candidate Architecture

- Famiglia: tiled_systolic_transformer
- Tile: 32x32
- Tile count: 25
- Mesh logica: 160x160
- PE count: 25600
- SRAM per tile: 640 KB
- Global buffer: 6 MB
- Bus width: 1536 bit
- Throughput stimato: 58.88 TOPS

## Rationale

- Candidato valutato: throughput_max.
- Famiglia architetturale scelta: tiled_systolic_transformer.
- Numero di MAC stimato con formula throughput / (2 * frequenza), pari a circa 21740 unita'.
- Dimensionamento iniziale in mesh 160x160 con tile 32x32.
- Stimato throughput teorico di 58.88 TOPS a 1150 MHz.
- Potenza stimata iniziale: 37.3 W.
- Area stimata iniziale: 5.5 mm2.
- Batch massimo 4: aumento buffer locale e ampiezza del fabric.
- Profilo throughput_max: privilegio frequenza e banda rispetto al costo energetico.

## Planned Modules

- mac_unit
- processing_element
- systolic_tile
- scratchpad_controller
- tile_compute_unit
- dma_engine
- scheduler
- top_npu
- prefetch_buffer
