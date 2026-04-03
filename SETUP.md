# Setup for ForgeNPU

Questo file descrive cosa serve per eseguire `ForgeNPU` su una macchina target diversa da quella attuale.

## Obiettivo

La macchina target deve poter fare quattro cose:

1. eseguire la pipeline Python del progetto;
2. generare artifact RTL e testbench;
3. lanciare tool EDA per lint, simulazione e sintesi;
4. opzionalmente usare un backend LLM per generazione guidata.

## Requisiti Minimi

- `git`
- `python3 >= 3.9`
- `pip`
- shell Unix-like (`bash` o `zsh`)

## Tool EDA Richiesti

Per chiudere davvero il loop automatico servono:

- `iverilog` per compilazione e simulazione
- `verilator` per lint
- `yosys` per sintesi

Tool opzionali per step successivi:

- `openroad`
- `openlane`

## Dipendenze Python

Il progetto corrente usa quasi solo la standard library. Per il backend LLM opzionale serve anche:

- package `openai`

## Installazione Consigliata

### macOS con Homebrew

```bash
brew install python
brew install icarus-verilog
brew install verilator
brew install yosys
python3 -m pip install --upgrade pip
python3 -m pip install openai
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y python3 python3-pip git iverilog verilator yosys
python3 -m pip install --upgrade pip
python3 -m pip install openai
```

## Bootstrap del Progetto

```bash
git clone https://github.com/mdomina/ForgeNPU.git
cd ForgeNPU
python3 -m unittest discover -s tests
python3 -m create_npu.cli doctor
```

## Variabili Ambiente LLM

Servono solo se vuoi attivare il backend LLM:

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="<optional-base-url>"
export CREATE_NPU_ENABLE_LIVE_LLM=1
```

Note:

- senza `OPENAI_API_KEY`, la pipeline ripiega sul backend euristico;
- senza `CREATE_NPU_ENABLE_LIVE_LLM=1`, il sistema non tenta chiamate live;
- con `--generator-backend llm`, il progetto salva comunque `llm_request.json` per debugging e replay.

## Verifica Installazione

Dopo il setup, controlla:

```bash
python3 -m create_npu.cli doctor --generator-backend llm --llm-model gpt-test
python3 -m create_npu.cli run \
  --requirement "Voglio una NPU INT8 da 50 TOPS con supporto transformer e batch 1-4." \
  --num-candidates 3
```

## Risultato Atteso

Se la macchina e' pronta:

- `doctor` deve mostrare `iverilog`, `verilator` e `yosys` come disponibili;
- la run deve creare `runs/output_<timestamp>/` con `result.json`, `candidates.json` e i log dei tool;
- il `python_reference` deve passare;
- se i tool EDA sono installati, anche lint/simulazione/sintesi possono essere eseguiti davvero.

## Consiglio Operativo

Per ambienti ripetibili conviene usare una di queste opzioni:

- una VM Linux dedicata al progetto;
- un container Docker con Python + tool EDA preinstallati;
- una macchina CI che esegue `doctor`, test e una run campione ad ogni modifica.
