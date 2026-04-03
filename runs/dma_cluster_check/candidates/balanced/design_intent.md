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
- SRAM per tile: 512 KB
- Global buffer: 6 MB
- Bus width: 1024 bit
- Throughput stimato: 51.20 TOPS

## Rationale

- Candidato valutato: balanced.
- Famiglia architetturale scelta: tiled_systolic_transformer.
- Numero di MAC stimato con formula throughput / (2 * frequenza), pari a circa 25000 unita'.
- Dimensionamento iniziale in mesh 160x160 con tile 32x32.
- Stimato throughput teorico di 51.20 TOPS a 1000 MHz.
- Potenza stimata iniziale: 25.1 W.
- Area stimata iniziale: 4.7 mm2.
- Batch massimo 4: aumento buffer locale e ampiezza del fabric.
- Profilo balanced: compromesso iniziale tra throughput, banda e consumi.

## Planned Modules

- mac_unit
- processing_element
- systolic_tile
- scratchpad_controller
- tile_compute_unit
- dma_engine
- scheduler
- top_npu
