# Design Intent

## Requirement

Voglio una NPU INT8 da 10 TOPS per dense GEMM.

## Parsed Spec

- Precisione: INT8
- Throughput target: 10.0 TOPS
- Potenza: None W
- Workload: dense_gemm
- Batch: 1-1
- Interfacce: AXI4, DMA, scratchpad_sram
- Tecnologia: generic_5nm
- Frequenza: 1000.0 MHz

## Assunzioni

- Batch non specificato: assumo batch 1.
- Interfacce non dichiarate: assumo AXI4, DMA e scratchpad SRAM.
- Nodo tecnologico non specificato: uso una baseline generica 5nm ASIC.
- Frequenza target non specificata: assumo 1000 MHz per la stima iniziale.

## Ambiguita'

- Budget di potenza assente o non parsabile.
- Nodo tecnologico assente.
- Frequenza target assente.

## Candidate Architecture

- Famiglia: tiled_systolic_array
- Tile: 16x16
- Tile count: 20
- Mesh logica: 80x64
- PE count: 5120
- SRAM per tile: 256 KB
- Global buffer: 5 MB
- Bus width: 512 bit
- Throughput stimato: 10.24 TOPS

## Rationale

- Candidato valutato: balanced.
- Famiglia architetturale scelta: tiled_systolic_array.
- Numero di MAC stimato con formula throughput / (2 * frequenza), pari a circa 5000 unita'.
- Dimensionamento iniziale in mesh 80x64 con tile 16x16.
- Stimato throughput teorico di 10.24 TOPS a 1000 MHz.
- Potenza stimata iniziale: 20.0 W.
- Area stimata iniziale: 1.1 mm2.
- Profilo balanced: compromesso iniziale tra throughput, banda e consumi.

## Planned Modules

- mac_unit
- processing_element
- systolic_tile
- scratchpad_controller
- dma_engine
- scheduler
- top_npu
