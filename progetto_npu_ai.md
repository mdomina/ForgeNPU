# Progetto: IA per Generare il Codice di una NPU da Requisiti in Linguaggio Naturale

## Visione

Realizzare un sistema di IA capace di trasformare una richiesta in linguaggio naturale, ad esempio:

> "Voglio 1000 TFLOPS di potenza in INT8"

in una proposta tecnica completa per una NPU, fino alla generazione del codice hardware dei moduli principali in Verilog/SystemVerilog, con verifica automatica tramite simulatori e tool EDA.

L'obiettivo non e' solo generare RTL, ma costruire un agente che:

- capisca il requisito espresso in linguaggio naturale;
- lo traduca in vincoli architetturali concreti;
- generi i blocchi hardware coerenti con tali vincoli;
- verifichi automaticamente correttezza, area, frequenza e throughput;
- migliori nel tempo grazie al feedback dei tool di verifica e sintesi.

## Problema da Risolvere

Oggi la progettazione di acceleratori richiede:

- traduzione manuale dei requisiti di prodotto in specifiche hardware;
- definizione dell'architettura della NPU;
- scrittura dei moduli RTL;
- verifica funzionale;
- sintesi e ottimizzazione iterativa.

Il progetto punta ad automatizzare gran parte di questo processo usando un'IA specializzata nella generazione di hardware.

## Obiettivo Principale

Dato un input testuale ad alto livello, il sistema deve produrre:

1. una specifica tecnica strutturata;
2. un'architettura candidata della NPU;
3. il codice RTL dei moduli principali;
4. testbench e casi di verifica;
5. report automatici su correttezza e qualita' del design.

## Esempio di Flusso

Input utente:

```text
Voglio 1000 TFLOPS di potenza in INT8, supporto per matrici dense, inferenza batch 1-16, consumo entro 250 W.
```

Output atteso:

- assunzioni esplicite mancanti, ad esempio frequenza target, tecnologia, memoria disponibile;
- proposta architetturale, ad esempio array sistolico, numero di PE, gerarchia memoria, larghezza bus;
- specifica dei moduli;
- codice RTL dei singoli blocchi;
- risultati di simulazione, sintesi e stima prestazioni.

## Principio Fondamentale

L'IA non deve essere valutata principalmente da un umano, ma dai tool di progettazione.

Il ciclo base e':

1. l'IA genera un modulo RTL;
2. il modulo viene controllato con lint e simulazione;
3. il modulo viene sintetizzato;
4. i risultati diventano un punteggio;
5. il punteggio guida le iterazioni successive.

Questo approccio e' il cuore del progetto: "reinforcement learning from design feedback".

## Architettura del Sistema

### 1. Requirement Interpreter

Componente che converte il linguaggio naturale in una specifica strutturata.

Esempi di campi:

- precisione numerica: INT8, FP16, BF16;
- throughput target: TOPS o TFLOPS;
- potenza massima;
- latenza massima;
- area target;
- tipo di workload: GEMM, conv, transformer, sparse, dense;
- interfacce: AXI, SRAM locale, DMA;
- nodo tecnologico o FPGA target.

Se l'input e' ambiguo, il sistema deve:

- fare domande di chiarimento;
- oppure applicare assunzioni standard e dichiararle esplicitamente.

### 2. NPU Architect

Componente che traduce la specifica in una microarchitettura.

Output tipici:

- numero di Processing Elements;
- dimensione dell'array;
- datapath width;
- schema di accumulo;
- memoria locale e tiling;
- network on chip o bus interni;
- frequenza di progetto;
- stima teorica di throughput.

### 3. RTL Generator

Genera il codice dei moduli hardware, ad esempio:

- `mac_unit`
- `processing_element`
- `systolic_array`
- `accumulator_bank`
- `activation_unit`
- `scratchpad_controller`
- `dma_engine`
- `scheduler`
- `top_npu`

### 4. Verification Harness

Pipeline automatica che esegue:

- lint del codice;
- compilazione;
- simulazione con testbench;
- confronto con golden model software;
- sintesi logica;
- eventualmente place and route e timing analysis.

Tool candidati:

- Verilator
- Icarus Verilog
- Yosys
- OpenROAD / OpenLane

### 5. Design Scorer

Converte i risultati tecnici in uno score numerico.

Esempio di logica:

- se il codice non compila: score negativo;
- se compila ma fallisce i test: score basso;
- se passa i test: score base positivo;
- se area, frequenza e consumo sono vicini al target: bonus;
- se il design e' troppo grande o troppo lento: penalita'.

### 6. Learning Loop

Componente che usa lo score per migliorare il sistema.

Fasi possibili:

- best-of-N generation;
- rejection sampling;
- dataset building con esempi buoni e cattivi;
- supervised fine-tuning;
- reinforcement learning con reward automatico.

## Flusso Operativo del Progetto

### Fase 1: MVP

Obiettivo: generare e verificare singoli moduli semplici.

Scope:

- adder
- multiplier
- MAC unit
- FIFO
- SRAM controller base

Capacita' richieste:

- generazione RTL da prompt;
- lint e simulazione automatici;
- punteggio automatico;
- archivio dei tentativi.

### Fase 2: Moduli NPU Reali

Obiettivo: generare blocchi significativi di una NPU.

Scope:

- processing element INT8;
- catena MAC + accumulo;
- systolic tile;
- controller del dataflow;
- memoria locale e load/store path.

Capacita' richieste:

- verifica contro modello Python;
- sintesi con Yosys;
- prime metriche di area e frequenza.

### Fase 3: Top-Level Assembly

Obiettivo: comporre i moduli in una NPU completa.

Scope:

- integrazione dei blocchi;
- interfacce di memoria;
- scheduler;
- controllo della pipeline;
- configurabilita' del tile size.

Capacita' richieste:

- generazione gerarchica;
- verifica end-to-end;
- stima del throughput architetturale.

### Fase 4: Learning from Design Feedback

Obiettivo: far migliorare il sistema automaticamente.

Strategia:

- generare piu' candidate per la stessa specifica;
- misurarle con tool EDA;
- tenere solo le migliori;
- usare i risultati per riaddestrare o ottimizzare il modello.

## Roadmap Esecutiva Dettagliata

Questa roadmap traduce le fasi del progetto in step concreti di sviluppo, in ordine di esecuzione.

### Step 0: Bootstrap del Repository

Obiettivo:

- creare la struttura base del progetto;
- definire package, CLI, directory artifact e documentazione iniziale;
- rendere possibile una prima run locale end-to-end.

Output atteso:

- package Python del progetto;
- comando CLI;
- directory `runs/`;
- file README e setup.

### Step 1: Requirement Parsing

Obiettivo:

- convertire il linguaggio naturale in una specifica strutturata;
- estrarre throughput, precisione, batch, workload, potenza, frequenza e tecnologia;
- esplicitare assunzioni e ambiguita'.

Output atteso:

- parser robusto dei requisiti;
- modello dati per la specifica;
- output JSON leggibile e persistente.

### Step 2: Candidate Architecture Planning

Obiettivo:

- generare piu' architetture candidate per la stessa specifica;
- stimare numero di PE, dimensione tile, SRAM locale, banda, frequenza, area e potenza;
- selezionare il candidato migliore con uno score iniziale.

Output atteso:

- candidati `balanced`, `throughput_max`, `efficiency`;
- archivio dei candidati;
- stima architetturale oggettiva.

### Step 3: Seed RTL dei Blocchi Aritmetici

Obiettivo:

- generare moduli elementari verificabili;
- creare testbench e casi di verifica minimali;
- costruire il primo loop automatico di scoring.

Scope:

- `adder`
- `multiplier`
- `mac_unit`

Output atteso:

- RTL seed dei moduli;
- testbench SystemVerilog;
- golden model software di riferimento.

### Step 4: Processing Element

Obiettivo:

- costruire un `processing_element` riusabile a partire dal MAC;
- supportare controllo `compute_en`, bypass e clear dell'accumulo;
- verificarne il comportamento con testbench e golden model.

Output atteso:

- modulo `processing_element`;
- casi di test signed e control-path;
- integrazione nel harness.

### Step 5: Systolic Tile

Obiettivo:

- comporre piu' `processing_element` in un blocco gerarchico reale;
- creare un primo `systolic_tile` parametrico e verificabile;
- validare il comportamento del tile con modello Python dedicato.

Output atteso:

- modulo `systolic_tile`;
- testbench del tile;
- casi di verifica del tile nel golden model.

### Step 6: Dataflow e Memoria Locale

Obiettivo:

- introdurre controllo del flusso dati e moduli memoria locali;
- implementare `scratchpad_controller` e primi path di load/store;
- collegare tile e memoria con interfacce chiare.

Output atteso:

- `scratchpad_controller`;
- primitive di buffering locale;
- test dei percorsi dati.

### Step 7: Scheduler e Top-Level NPU

Obiettivo:

- costruire una NPU minima assemblando tile, memoria e controllo;
- introdurre `scheduler`, configurazione di tile size e path top-level;
- fornire una prima stima end-to-end di throughput.

Output atteso:

- `scheduler`;
- `top_npu`;
- test di integrazione gerarchica.

### Step 8: Verification Harness Completo

Obiettivo:

- strutturare una pipeline a strati per verificare rapidamente i design;
- usare prima golden model e lint, poi simulazione, poi sintesi;
- salvare log e risultati come artifact permanenti.

Output atteso:

- harness con `python_reference`, `verilator`, `iverilog`, `yosys`;
- log persistenti;
- diagnostica dell'ambiente e dello stato toolchain.

### Step 9: Scoring e Ranking

Obiettivo:

- trasformare i risultati tecnici in un punteggio comparabile;
- pesare correttezza funzionale, area, frequenza, throughput e consumo;
- scegliere il miglior candidato e conservare il ranking storico.

Output atteso:

- funzione di scoring stabile;
- ranking dei candidati;
- archivio dei tentativi e del best candidate.

### Step 10: Backend LLM per RTL Generation

Obiettivo:

- affiancare al generatore euristico un backend LLM specializzato;
- produrre prompt strutturati e artifact riproducibili;
- mantenere fallback locale pulito in assenza di configurazione LLM.

Output atteso:

- supporto `--generator-backend llm`;
- `llm_request.json`;
- passaggio graduale da seed statici a generazione guidata.

### Step 11: Dataset Building

Obiettivo:

- salvare per ogni run requisito, specifica, architettura, RTL, testbench, log e score;
- costruire un dataset proprietario di esempi buoni e cattivi;
- rendere possibile ranking offline e analisi degli errori.

Output atteso:

- schema dati stabile;
- raccolta strutturata dei tentativi;
- base per supervised fine-tuning o rejection sampling.

### Step 12: Learning from Design Feedback

Obiettivo:

- usare i punteggi dei tool come reward automatico;
- migliorare progressivamente la qualita' dei moduli generati;
- chiudere il ciclo IA -> RTL -> tool EDA -> score -> miglioramento.

Output atteso:

- best-of-N automatico;
- filtro dei candidati peggiori;
- pipeline pronta per RL o ottimizzazione iterativa.

### Step 13: Industrializzazione del Flusso

Obiettivo:

- rendere il sistema eseguibile in ambienti ripetibili e CI;
- definire setup standard, container e pipeline di validazione;
- preparare il progetto a workload piu' grandi e team multipli.

Output atteso:

- setup documentato;
- esecuzione in CI o macchina dedicata;
- processo riproducibile per benchmark e regressioni.

## Struttura dei Dati

Ogni sample del dataset dovrebbe contenere:

- requisito in linguaggio naturale;
- specifica strutturata derivata;
- architettura proposta;
- codice RTL generato;
- testbench;
- log di lint;
- log di simulazione;
- report di sintesi;
- metriche finali;
- score complessivo.

Questo permette di costruire un dataset proprietario ad alto valore.

## Metriche di Successo

Le metriche principali del progetto devono essere oggettive.

### Metriche di correttezza

- percentuale di moduli che compilano;
- percentuale di moduli che passano i testbench;
- percentuale di equivalenza con il golden model.

### Metriche di qualita' hardware

- area sintetizzata;
- frequenza massima stimata;
- throughput stimato;
- efficienza energetica;
- utilizzo memoria e banda.

### Metriche di produttivita'

- tempo medio per ottenere un modulo valido;
- numero medio di iterazioni per convergere;
- percentuale di richieste completate senza intervento umano.

## Rischi Principali

### 1. Ambiguita' dei requisiti

Una frase come "voglio 1000 TFLOPS in INT8" non basta da sola. Mancano dettagli come:

- processo tecnologico;
- budget energetico;
- frequenza target;
- dimensione del chip;
- memoria disponibile;
- workload reale.

Il sistema deve quindi gestire l'ambiguita', non ignorarla.

### 2. Codice plausibile ma sbagliato

Gli LLM possono scrivere RTL che sembra corretto ma non lo e'. Per questo la verifica automatica e' obbligatoria.

### 3. Costo computazionale

Sintesi e place-and-route sono lenti. Il loop deve essere stratificato:

- filtro veloce con lint;
- poi simulazione;
- poi sintesi;
- solo infine timing e PnR sui migliori candidati.

### 4. Reward mal definito

Se il punteggio e' costruito male, il modello imparera' ad ottimizzare la metrica sbagliata. La correttezza funzionale deve avere priorita' assoluta.

## Strategia Tecnica Consigliata

Per partire in modo realistico:

1. usare un LLM coder specializzato su Verilog/SystemVerilog;
2. limitare inizialmente il problema a moduli singoli;
3. costruire un harness robusto con Verilator + Yosys;
4. creare un golden model Python per i blocchi aritmetici;
5. raccogliere dataset di tentativi, successi e fallimenti;
6. solo dopo introdurre reinforcement learning vero e proprio.

## Output Atteso del Sistema

Per ogni richiesta utente, il sistema ideale deve produrre:

- una specifica tecnica leggibile;
- le assunzioni effettuate;
- una proposta architetturale;
- il codice dei moduli;
- i testbench;
- i report di verifica;
- una valutazione finale di fattibilita'.

## Caso d'Uso Finale

Utente:

```text
Voglio una NPU INT8 da 1000 TFLOPS ottimizzata per inferenza transformer, con consumo massimo di 250 W e supporto batch 1.
```

Sistema:

1. chiarisce le assunzioni mancanti;
2. propone una microarchitettura;
3. genera i moduli RTL;
4. lancia verifica e sintesi;
5. restituisce una soluzione con metriche misurate;
6. itera automaticamente per migliorare il design.

## Deliverable del Progetto

- motore di parsing dei requisiti;
- generatore di specifiche strutturate;
- generatore RTL dei moduli NPU;
- framework automatico di verifica;
- sistema di scoring;
- dataset interno di design hardware;
- ciclo di apprendimento da feedback EDA.

## Sintesi

Questo progetto mira a costruire un'IA ingegneristica, non un semplice generatore di codice.

Il valore reale non sta solo nel produrre Verilog, ma nel chiudere il ciclo:

- capire il requisito,
- proporre l'architettura,
- generare il design,
- verificarlo automaticamente,
- migliorarlo con feedback misurabile.

Se realizzato bene, il sistema puo' diventare una piattaforma per progettare acceleratori hardware in modo molto piu' rapido, iterativo e scalabile rispetto al flusso manuale tradizionale.
