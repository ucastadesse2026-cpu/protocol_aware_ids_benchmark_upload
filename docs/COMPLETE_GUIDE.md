# Complete step-by-step guide: from a fresh Anaconda Prompt to a finished paper

Everything is typed in **Anaconda Prompt** (Start menu -> "Anaconda Prompt") unless a step says *Linux*. Lines starting
with `#` are comments. Replace `D:\work\protocol_aware_ids_benchmark` with your project folder. The path may contain
spaces: put it in quotes after `cd /d`.

**Rules that keep the paper honest** (they are enforced by the tools where possible):
1. Only numbers that appear in a table file written by `make_tables.py` from real runs go into the manuscript.
2. Anything stamped `SYNTHETIC-DATA` is a plumbing check, never a result.
3. Decision thresholds (Table 9) are fixed and committed **before** the first real run.
4. Report outcomes as measured, including when H-blind or any research-question hypothesis is not supported.
5. State clearly where the satellite traffic comes from (NOS3 simulator with author-scripted attacks, not flight data).

**Tested status.** Passed 77 unit/end-to-end tests on Linux; the PyTorch tests passed on your Windows machine (the
AutoAttack wrapper was fixed afterwards and not re-run there). **Never executed on a real NOS3 install:** the NOS3
tools in Part 5. **Never executed on Windows:** the `.bat` files (the Python tools are cross-platform).

---------------------------------------------------------------------------------------------------------------------
## Part 0. The map

| Part | What | Where | Needs | Done when |
|---|---|---|---|---|
| 1 | Environment, update, unit tests | Windows | - | tests print `OK` |
| 2 | Smoke run (synthetic) | Windows | Part 1 | tables with SYNTHETIC banner exist |
| 3 | Download 3 terrestrial datasets | Windows (browser) | - | `--check` shows 3 x `[ok]` |
| 4 | Pilot (real terrestrial, synthetic satellite) | Windows | Parts 1-3 | pilot finishes, timings noted |
| 5 | Satellite traffic from NOS3/cFS | **Linux** + Windows | a Linux machine | converted stream + profile |
| 6 | Pre-register (freeze config) | Windows | Part 5 | config committed |
| 7 | Full experiments | Windows | Parts 3, 5, 6 | `results_final` exists |
| 8 | Check results | Windows | Part 7 | checklist passed |
| 9 | Write the results into the manuscript | doc | Part 8 | every TBD replaced by a measured value |
| 10 | Finalise, release, submit | Windows/doc | Part 9 | submission package |
| 11 | Troubleshooting | - | - | - |

Do not skip a gate: if a "Done when" is not met, send me the output instead of moving on.

---------------------------------------------------------------------------------------------------------------------
## Part 1. Environment, update, unit tests

### 1.1 First time only: create the environment
```bat
conda create -n ids python=3.11 -y
conda activate ids
conda install -y numpy pandas scipy scikit-learn pyyaml matplotlib
pip install torch adversarial-robustness-toolbox
python -c "import numpy, pandas, scipy, sklearn, yaml, matplotlib, torch, art; print('torch', torch.__version__, '| art', art.__version__)"
```
(Alternative: `conda env create -f environment.yml`.) The code runs on **CPU only**; no GPU is used.
Open a new Anaconda Prompt later with `conda activate ids` before anything else.

### 1.2 Install or update the project
Put `protocol_aware_ids_benchmark.zip` next to (not inside) the project folder, then:
```bat
cd /d "D:\your\parent\folder"
tar -xf protocol_aware_ids_benchmark.zip
cd protocol_aware_ids_benchmark
rmdir /s /q results\cache
```
`tar` overwrites code files and keeps everything not in the zip: your datasets (`data\raw`), your own configs and the
`ids` environment. The `rmdir` deletes cached models (it may say the folder is missing: fine). If you edited
`configs\default.yaml` directly, copy it to `configs\my_run.yaml` first.

### 1.3 Run the unit tests
```bat
python -m unittest discover -s tests
```
Expected: `Ran 78 tests ... OK` (0 skipped with torch and ART installed). Takes about 1-2 minutes. If anything fails,
send me the last 40 lines and the name of the failing test; do not continue.

---------------------------------------------------------------------------------------------------------------------
## Part 2. Smoke run (about 1 minute, synthetic data)
```bat
run_all.bat configs\smoke.yaml
dir results_smoke\tables
type results_smoke\tables\table9_rq_summary.md
```
Expected: the five scripts finish, `results_smoke\tables` holds table3...table9, `results_smoke\figures` holds
fig3-fig5, and every table starts with **SYNTHETIC-DATA BANNER**. This only proves the pipeline runs. If a script fails,
`run_all.bat` names the log in `logs\`.

---------------------------------------------------------------------------------------------------------------------
## Part 3. Terrestrial datasets
```bat
python data\download_datasets.py --help
```
The help lists the official pages (checked 2026-10-10). Place the files exactly here:

| Dataset | How to get it | Save as |
|---|---|---|
| NSL-KDD | official page says "no longer available"; third-party GitHub mirror (not official): run the command below | `data\raw\nsl_kdd\KDDTrain+.txt` |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset -> the "HERE" link opens a UNSW SharePoint folder; download in a browser; use the **training-set** CSV (check its size against the provider's description) | `data\raw\unsw_nb15\UNSW_NB15_training-set.csv` |
| CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html -> registration form at http://cicresearch.ca/CICDataset/CIC-IDS-2017/ ; download `MachineLearningCSV.zip`, extract all CSVs | `data\raw\cicids2017\*.csv` |

```bat
python data\download_datasets.py --fetch nsl_kdd --url https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/KDDTrain%2B.txt
python data\download_datasets.py --check
python data\preprocess.py --config configs\default.yaml --dataset nsl_kdd
python data\preprocess.py --config configs\default.yaml --dataset unsw_nb15
python data\preprocess.py --config configs\default.yaml --dataset cicids2017
```
Gate: `--check` shows three `[ok]`, and each preprocess line prints `real` and three shapes (train/val/test).
NSL-KDD should show 4 features (core view), the other two 8. Known facts from your machine: CICIDS2017 loads as
300000/100000/100000 rows (the 500,000-row cap, 60/20/20); your UNSW file produced 82,332 rows in total, which is the
size of the provider's *testing* partition, so confirm you have the partition you intend (to enlarge the sample, join
both partition CSVs; their columns match). Cite the dataset papers (refs [62], [63], [64] in the manuscript).
**Official UNSW-NB15 partitions:** to train on the official training partition and test on the official testing partition,
download both CSVs and add `test_path` to the dataset entry of YOUR config (not to default.yaml):
```yaml
data:
  terrestrial:
    unsw_nb15: {path: data/raw/unsw_nb15/UNSW_NB15_training-set.csv, test_path: data/raw/unsw_nb15/UNSW_NB15_testing-set.csv, source: real}
```
(the training file is then split 75/25 into train/validation; the whole testing file is the terrestrial test set).

---------------------------------------------------------------------------------------------------------------------
## Part 4. Pilot: real terrestrial data, synthetic satellite data
```bat
run_all.bat configs\pilot.yaml
type results_pilot\tables\table3_clean_terrestrial.md
```
`configs\pilot.yaml` uses seed 1, 5 CNN epochs, random forest + CNN, CICIDS2017 capped at 100,000 rows. All outputs carry
the synthetic banner (the satellite side is synthetic); only terrestrial rows are meaningful, and even those are pilot
numbers (5 epochs). **Write down the elapsed time of each script** (printed in `logs\*.txt`, or time them with
`run_seeds.py` in Part 7). Rough planning rule, not a prediction: full time is about
pilot time x 5 (seeds) x (50/5 for the neural-training part) x (up to 5 for CICIDS2017, if you keep 500,000 rows). Cut
the cost by lowering `max_rows`, epochs, `attacks.max_windows`, or the number of models; record every reduction in the
paper. Gate: the pilot finishes without `FAILED`; if it fails, send me the failing log's last 30 lines.

---------------------------------------------------------------------------------------------------------------------
## Part 5. Satellite-domain traffic from NOS3 / cFS

What the paper may call this data: **NOS3/cFS simulator traffic with author-scripted attacks and ground-truth
interval labels**. It is not flight or operational data. Decide first:

* **Route A (recommended, needs a Linux machine):** NOS3 + scripted traffic, steps 5.1-5.8.
* **Route B (no Linux machine / no real satellite data):** you can still run everything with synthetic satellite
  windows, but then **no satellite-robustness claim can be made from real traffic**: the title, abstract and
  contributions must be reduced to a benchmark-framework paper with terrestrial results, and every satellite-domain
  number must be labelled synthetic. Do Parts 6-10 with `configs\default.yaml` and `data.satellite.source: synthetic`.
* **Route C:** any other real packet capture you obtain: convert it to the stream format (`data\README.md`) and adapt
  the parser profile.
* **Route D (strengthens Route A): reproduce the published CuCD-ID scenarios.** CuCD-ID (Fayyaz, Yang, El-Khatib, Data
  in Brief 2026, doi 10.1016/j.dib.2026.112598; data: Mendeley Data doi 10.17632/7n2d42pm3n.3) was generated with the
  same NOS3/cFS stack (OpenC3 COSMOS v4) and ships **the COSMOS scripts that reproduce its traffic** for five
  scenarios (normal, command flooding, GPS data injection, defence impairment via Limit Checker/Checksum manipulation,
  storage exhaustion). Running *their* scripts while `tcpdump` captures, instead of my generic generator, gives packet
  captures whose attack scenarios are published and citable. I have **not** seen the scripts or their licence (the
  Mendeley page I could read did not show it): download them, read how each scenario sends traffic, and check that
  the attacks you want travel through the cFS command-ingest UDP port (only those yield command packets the conformant
  attacker can edit). Record each scenario's start/end times in `attacks.csv` (columns `start,end,attack_type`).

**What the two satellite datasets actually are (checked 2026-10-11 against the providers' pages):**

| Dataset | Verified facts | Fit with this benchmark |
|---|---|---|
| **UNSW-IoTSAT** (Abdelhameed, Turnbull, Koroniotis, *Cyber Security and Applications* vol. 4, 100133, 2026 - citation as listed in a third-party paper; repo https://github.com/Osama-Abdelhameed/UNSW-IoTSAT, MIT; data on a UNSW SharePoint folder linked from the repo README) | Hybrid testbed (real sensors, real CCSDS-framed BPSK radio, simulated orbit); 404,798 records, 109 features + 3 CCSDS companion columns (frame count, 14-bit packet sequence count, APID) that are **derived deterministically from the attack labels**; six attack types (jamming, spoofing, replay, DoS, MitM, eavesdropping). | **Tabular, not packets.** No packet/byte counts, so the common features Fc cannot be computed from rows, and packet-level conformant attacks cannot run on it. Usable at most as a feature-level supplement with its own feature set (a different study design). |
| **CuCD-ID** (doi above) | NOS3/cFS digital twin; 25,000 raw rows x 31 columns + 22,465 x 23 noised; CCSDS header fields and 20-second window traffic statistics; five scenarios; COSMOS reproduction scripts included. | **Tabular, not packets** - but the scripts make Route D possible. Whether its window statistics can be mapped to Fc depends on its column list (send it to me). |

If you want either table used as satellite-domain evidence, tell me and send the column names: I will check whether
a defensible mapping to the common features exists before any code is written. Do not describe them in the paper as a
source of packet-level adversarial examples.

**What the sources confirm about NOS3** (nos3.readthedocs.io, github.com/nasa/nos3): Linux with Docker or a Linux VM
(Vagrant/VirtualBox route in github.com/nasa-itc/deployment); the install page says 32 GB RAM may run it and 64 GB+
is recommended; `make launch` / `make stop`; ground link = cFS CI (command ingest) and TO (telemetry output) apps over
UDP; all packets are CCSDS with the secondary header on; send `TO_ENABLE_OUTPUT` for telemetry via the TO path; COSMOS,
YAMCS, F Prime and AIT ground systems are supported. cFS commands = 6-octet primary header + 1-octet function code +
1-octet XOR checksum + payload (no timestamp, no CRC). **Not verified:** UDP port numbers, the location of the COSMOS
command definitions, the telemetry secondary-header length (assumed 6), the checksum initial value (assumed 0xFF; the
tools measure it).

### 5.0 Command-line (headless) server: use the dedicated guide
If your NOS3 machine has **no desktop** (SSH only), follow **`docs/NOS3_HEADLESS_GUIDE.md`** instead of 5.2-5.6: it covers
install, headless launch, port discovery, harvesting real commands, the capture session and fetching the files, with a
gate after every step. Then continue at 5.7 here. (New helper tools: `tools/pcap_flows.py` finds the command and telemetry
ports from a capture; `tools/catalog_from_stream.py` builds the command catalog from real captured commands.)

### 5.1 Pick the machine (decision gate)
NOS3 does not run natively on Windows; I have not verified WSL2. Use a Linux PC/server or a Linux VM with enough RAM
(see above). Tell me your RAM and OS if unsure. Only the capture file travels back to Windows.

### 5.2 Install and start NOS3 (*Linux*)
Follow "Scenario - Installation" at https://nos3.readthedocs.io (steps change between versions). In essence:
`git clone https://github.com/nasa/nos3.git`, `cd nos3`, prepare/build per the docs, `make launch`. In COSMOS send a
NOOP and confirm telemetry arrives. Stop with `make stop`. Copy the benchmark folder (or just `tools\`) to the same
Linux machine and install Python 3.10+ with `numpy` there (the generator needs only the standard library plus
`data/ccsds_parser.py`; run `pip install numpy`).

### 5.3 Find the UDP ports (*Linux*)
```bash
sudo ss -u -a -n -p | head -40
sudo tcpdump -i any -n -c 60 udp
```
Send a NOOP from COSMOS' Command Sender while `tcpdump` runs. **Command port** = UDP *destination* port of the packet
that appears (ground -> cFS CI). **Telemetry port** = UDP *destination* port of the steady stream in the other
direction (cFS TO -> ground); if there is none, send `TO_ENABLE_OUTPUT`. Write both down.

### 5.4 Build the command catalog (*Linux*)
```bash
cp tools/command_catalog.example.json tools/my_catalog.json
nano tools/my_catalog.json        # replace every TODO
```
For each command: `apid` (message ID & 0x7FF), `fc` (function code), `payload_hex`, and `klass`: `benign` (routine, e.g.
NOOP / request-HK) or `sensitive` (legal but unusual and **harmless in the simulator**). Sources: the COSMOS command
definition text files in the NOS3 repo (find with `grep -rl "COMMAND" --include=*.txt gsw/`; the path differs per
version) or send each command once while `sudo tcpdump -i any -n -X udp` shows the bytes: APID = low 11 bits of the first
16-bit word, function code = byte 7. Avoid resets/deletes/power commands. The generator refuses to run while TODOs remain.

### 5.5 Capture and generate (*Linux*, two terminals, same machine)
Terminal 1:
```bash
sudo tcpdump -i any -n -U -w nos3.pcap udp
```
Terminal 2 (leave NOS3 idle for a few minutes first so the start of the capture is attack-free):
```bash
python tools/nos3_traffic.py --host 127.0.0.1 --port <CMD_PORT> --catalog tools/my_catalog.json \
    --mode session --cycles 20 --benign-s 60 --attack-s 20 --intervals-out attacks.csv --i-own-this-simulator
```
Defaults: benign 1 cmd/s, flood 100/s, injection 2/s, recon 5/s (`--rates`), `--extra-payload-bytes` pads injection
commands. The tool only sends to loopback/private addresses (if NOS3 is in Docker, use the host/IP where the CI port is
reachable) and only with `--i-own-this-simulator`. Stop `tcpdump` (Ctrl-C) afterwards and copy `nos3.pcap` and
`attacks.csv` to Windows (D:\work\... any folder).

### 5.6 Convert the capture (Windows)
```bat
mkdir data\raw\nos3
python tools\pcap_to_stream.py --pcap D:\path\nos3.pcap --cmd-port <CMD_PORT> --tlm-port <TLM_PORT> --intervals D:\path\attacks.csv --out data\raw\nos3\nos3
```
It prints counts; `parse_errors` above ~5% means the wire format differs from your cFS build. A pcapng file (Wireshark)
must first be converted with `editcap -F pcap in.pcapng nos3.pcap`. Labels: command packets inside an attack interval =
attack (1); all telemetry and all other commands = benign (0). Duplicate datagrams from multi-interface capture are
removed (1 ms rule).

### 5.7 Derive and review the mission profile (Windows)
```bat
python tools\derive_profile.py --stream data\raw\nos3\nos3 --catalog tools\my_catalog.json
copy configs\nos3.yaml configs\my_nos3.yaml
notepad configs\my_nos3.yaml
```
Paste the printed `profile:` block into `configs\my_nos3.yaml` (the runs stop without it) and set `xor_init` to the value
the tool reports (255 or 0). Review: `max_user_data_*` is "observed maximum + 64 bytes": set your mission's real limit;
legal function codes come from the catalog/capture, so a legal command you never sent counts as illegal (conformant
attacks become *more* constrained: say so in the paper). Gate: the tool's split table shows **hundreds of windows and
attack windows in train, val and test**, and "already valid under this profile" is high; if not, add `--cycles` /
shorten periods and recapture.

### 5.8 Cross-check the data once
Confirm in the tool output: TC and TM counts are plausible, checksum validates ~100% for one initial value, attack
windows are not concentrated in one third of the capture. Keep `nos3.pcap`, `attacks.csv`, `my_catalog.json`, the
NOS3 commit hash (`git rev-parse HEAD` in the nos3 folder) and the exact generator command line: the paper needs them.

---------------------------------------------------------------------------------------------------------------------
## Part 6. Pre-register (freeze the configuration)
1. Open `configs\my_nos3.yaml` (or `my_run.yaml` for Route B). Check `datasets`, `model_families`, `seeds`,
   `data.satellite`, `profile`, and the whole `preregistration:` block (RQ thresholds from the manuscript).
2. Do **not** run anything real yet. Freeze with git (install once if needed: `conda install -y git`):
```bat
git init
git add configs tools\my_catalog.json
git commit -m "Frozen configuration and decision rules before the first real run"
git log -1 --format=%H
```
Record the printed commit hash. Any later change to thresholds must be a visible new commit and be disclosed in the paper.
Prove that nothing changed with `python tools\check_freeze.py --freeze <freeze-hash> --config configs\my_nos3.yaml`: it resolves `_base` inheritance at the freeze commit and now, and lists every differing setting (a `grep` on your own config proves nothing when the `preregistration:` block lives in `default.yaml`). A commit cannot contain its own hash: identify an amendment by a git tag (`git tag amendment-1`) or by `git log -- <file>`, and cite the tag/hash in the paper.
**If you change data, code or analyses after seeing results** (for example enlarging the capture), write `docs/AMENDMENT_TEMPLATE.md`
into a committed file BEFORE opening the new results: it records what changed, what did not, which run is primary and which
validity checks decide whether a number is informative. After unzipping a new code release over your project run
`git status` and `git diff -- configs`: configs that use `_base: default.yaml` inherit any change to `default.yaml`, so a
changed base config silently changes your frozen settings.
(Add your capture metadata too; do **not** commit raw datasets.)

---------------------------------------------------------------------------------------------------------------------
## Part 7. Full experiments (resumable, seed by seed)
```bat
python tools\run_seeds.py --config configs\my_nos3.yaml --seeds 1 2 3 4 5 --final-dir results_final
```
On a many-core machine add `--jobs 5` to run the five seeds in parallel (CPU only: the models are tiny and the GPUs are not used).
For each seed it runs the six scripts (baseline, standard attacks, ccsds attacks, in-domain reference, defenses, federated) into `results_s<seed>`, logs to `logs\`, prints elapsed time per script, stops at
the first failure with the log tail, and **skips finished seeds when you re-run the same command** (`--force` redoes
them). After the last seed it merges everything into `results_final` and builds the tables and figures.
To run on several machines: run `run_seeds.py` with different `--seeds` on each, copy the `results_s*` folders into one
place, then `python tools\merge_results.py --inputs results_s1 results_s2 ... --out results_final` and
`python make_tables.py --config configs\my_nos3.yaml --out-dir results_final`.
To rerun only some scripts for finished seeds (for example after a code fix): `python tools\run_seeds.py --config <cfg> --seeds 1 2 3 4 5 --only run_defenses --final-dir results_final`.
Run the five scripts by hand only for debugging: each takes `--config`, `--seeds`, `--out-dir`, `--no-cache`; each
overwrites its own CSV, so use a separate `--out-dir` per run.

---------------------------------------------------------------------------------------------------------------------
## Part 8. Check the results before writing anything
Open `results_final\tables\` and tick every item:
- [ ] No table starts with the SYNTHETIC-DATA banner (Route A/C). Route B: satellite numbers are synthetic, tables are stamped; keep them out of the headline claims.
- [ ] `table3`: clean terrestrial accuracy per model/dataset looks sane (not 50%, not 100.00% everywhere); the satellite control `table3b` shows the clean cross-domain behaviour.
- [ ] `table5`: **feasibility rate** is high (100% expected: infeasible candidates are rejected and counted as failed attacks); mean edits > 0; success rate not 0 everywhere.
- [ ] `table6`: G = G_domain + G_protocol holds (it is exact); look at the **sign** of G_protocol per model.
- [ ] `table7`: compare `paa` with `paa_clean_control`; if they are equal, the benefit is domain exposure, not adversarial content. τ shows "n/a" where terrestrial robustness gain is under 1 pp (by design).
- [ ] `table8`: federated G vs the centralised reference; check each K and alpha has rows.
- [ ] `table9`: each RQ's rule, observed values, outcome. The outcome words come from the thresholds you froze; do not edit them.
- [ ] Figures `fig3-fig5` opened and readable (no SYNTHETIC watermark).
- [ ] `logs\*.txt` contain no warnings you do not understand; seeds 1-5 are all present in `gap_per_seed.csv`.
**Red flags learned from a real run (check these every time):**
- [ ] `n_windows` in `ccsds_attacks.csv` (satellite test windows) is at least several hundred. With ~100 windows one accuracy has a binomial 95% half-width of about 9 pp, and seeds that change only the model do not capture that.
- [ ] Clean satellite accuracy (`table3b`) is not a near-all-attack predictor: FPR close to 100% means the terrestrial model flags everything, so "robustness" to attacks is meaningless for it (nothing left to evade).
- [ ] No row has accuracy UNDER ATTACK above its CLEAN accuracy (`table7`: Acc S ccsds vs Acc S clean). If so, the attack is not adaptive and the row is not evidence of robustness.
- [ ] tau is only meaningful when clean accuracy did not change; read `Acc S clean` and the `Clean->ccsds drop` columns before interpreting tau.
- [ ] Compare PAA with `paa_clean_control` (paired, with intervals) before claiming any benefit of adversarial content.
- [ ] `table6b` (exploratory worst-case conformant attack) next to `table6`: if the strongest attack is not the pre-registered one, report both and label 6b as post hoc.
- [ ] Feasibility below 100% in `table5`: run `python tools\diagnose_feasibility.py --config <cfg> --seed 1 --model cnn` to see which constraint fails.
- [ ] **Constant-predictor check:** compare each model's clean satellite accuracy with the always-attack and always-benign accuracies (the attack/benign base rates of the evaluated windows). A model within a few points of either, or with clean FPR >= 0.9, is a near-constant predictor and every robustness number measured on it is uninformative.
- [ ] Look at `table10_indomain_reference` (EXPLORATORY, detectors trained on satellite windows): it tells you whether benign and attack windows are separable in the shared features at all, and whether conformant attacks evade a detector that works.
- [ ] Scope matches the manuscript: datasets and model families that were NOT run must be removed from the title, abstract and Table 1.

### Part 8b. When in-domain detectors do not work: diagnose separability BEFORE changing anything
Learned from a real run: three of four detectors trained on the NOS3 training windows predicted "attack" for every test window
(accuracy equal to the attack base rate), and the fourth was below it. Two likely causes, both in the design and not in the
attacks: (1) the shared features (`core`/`extended`: durations, bytes, packet counts) contain **no APID or opcode information**, so
recon (APID sweep) and injection (rare opcodes) are invisible in them; (2) long windows (32 packets) **dilute sparse attacks** under
the label rule "any". Check it with the diagnostic (it never touches the test split):
```bat
python tools\diagnose_separability.py --config configs\my_nos3.yaml --seed 1
```
It prints the composition per split and attack type, label purity, Cohen's d per feature and attack type, a time-blocked CV of an
in-domain Random Forest, and a grid over feature view x window size x label rule scored on the **validation** split. Choose the design
from that table, write the choice into the amendment (commit it), then run the in-domain reference with it, for example:
```yaml
indomain: {view: sat_aug, window_size: 16, label_rule: any}      # in YOUR config; view sat_aug = shared features + APID entropy, sequence gaps, distinct APIDs
```
```bat
python tools\run_seeds.py --config configs\my_nos3.yaml --seeds 1 2 3 4 5 --jobs 5 --only run_indomain --final-dir results_ind_design2
```
Use a separate `--final-dir` per design so results are never mixed. A satellite-only view (`sat_aug`) is legitimate for in-domain detectors
but cannot be used for terrestrial-trained models (those features do not exist in terrestrial data): the cross-domain experiments keep
the shared features. Also rerun the other stages with `attacks.max_windows` large enough to cover the whole test split, so that every table
uses the same evaluation windows (a run that mixes a 565-window prefix with a 1,196-window full split cannot be compared across tables).

If the data show something surprising (e.g. 100% accuracy under attack, negative gaps everywhere), investigate with me
before interpreting it; many such results are bugs or data artefacts, and some are real findings.

---------------------------------------------------------------------------------------------------------------------
## Part 9. Writing the results into the manuscript
The manuscript is the tab "Manuscript v2 (69 refs, TBD results)" of the shared document. Work in this order:

| Manuscript part | What to do | Source file |
|---|---|---|
| Table 1 (datasets) | fill row counts (train/val/test after cleaning); NOS3/cFS row: NOS3 + cFS version, packet counts, attack classes; **delete rows for resources you did not use** | `derive_profile.py` output, `preprocess` printouts |
| Section IV Phase 1 | describe the satellite data as NOS3/cFS simulator traffic; replace the "project profile" secondary-header/CRC text with the **cFS format** (primary header + function code + XOR checksum; arrival time from capture); report window size, label rule "any", dedupe rule | `docs`, config |
| Section V | tools versions, hardware, seeds, final epochs/caps you actually used, any reductions | `environment_full.yml`, config |
| Tables 3-8 (Section VI) | copy each table's rows from the matching `.md` file; keep the same column order | `results_final\tables\table*.md` |
| Figures 3-5 | insert the PNG files; keep captions factual | `results_final\figures\*.png` |
| Table 9 | copy rows verbatim | `table9_rq_summary.md` |
| Sections VI A-H text | describe what the numbers show, per RQ, using the pre-registered rule wording; no adjectives beyond the data | tables |
| Section VII | rewrite "How the outcomes will be interpreted" as "what the outcomes mean": use the branch that matches the measured sign of G_protocol (H-blind, H-weak or mixed) and the PAA-vs-control comparison | tables |
| Abstract | replace the TBD sentence (template below) | tables |
| Section VIII | state what was found, what was not supported, remaining limitations | all |
| Limitations | add: simulator-only satellite data with scripted attacks, regular benign command traffic, TC-only attacker, shared-feature information loss, catalog-defined legal commands, host-CPU latency proxy | - |

**Abstract template** (fill brackets only with measured values and keep the intervals):
"Across [N] seeds and [datasets], terrestrial-trained models reached [x-y]% clean terrestrial accuracy but [a-b]% on
[NOS3 simulator] satellite traffic without any attack (clean cross-domain drop [d] pp). Under PGD the satellite-domain
accuracy was [..]%, and under CCSDS-conformant attacks [..]%, giving G_protocol of [m] pp (95% CI [lo, hi]) on average,
[supporting / not supporting / mixed for] H-blind [..]. Standard adversarial training transferred [tau] of its
terrestrial gain [..]; PAA recovered [..]% of G_protocol at [..]x training time, [versus ..% for the clean-augmentation
control]. Federated models deviated from the centralised gap by [..] pp for alpha >= 0.5 and [..] pp for alpha = 0.1."

**Sending it to me:** paste (or upload) `table3` ... `table9` (.md or .csv), `gap_per_seed.csv`, the printouts of
`derive_profile.py`, and the final config. I will fill Section VI, VII, VIII and the abstract strictly from them and
flag any claim they do not support. I cannot invent, smooth or "round into" a hypothesis.

**If an RQ is not supported:** say so plainly in Table 9, the abstract and Section VIII; discuss why (with the
tables as evidence), and keep the contribution as the benchmark and its measurements. A well-documented null or
mixed result is publishable; a result that was nudged toward the hypothesis is not.

---------------------------------------------------------------------------------------------------------------------
## Part 10. Finalise, release, submit
### 10.1 Facts to verify (the manuscript marks them with [brackets])
- [ ] Every reference's volume/issue/pages/DOI from the publisher record; fix the title wording flagged in [8] and [20].
- [ ] [3] and [43] are the same paper: keep one (68 references) or keep both only if the venue accepts it.
- [x] [68] González-Ramos et al. (J. Inf. Secur. Appl. 100, 104467, 2026, doi 10.1016/j.jisa.2026.104467) **verified** against its abstract: FGSM, PGD, AutoAttack and PGD-AT on a hybrid CNN-LSTM-Transformer NIDS, terrestrial data. It overlaps with your attack families and defense, so your distinction must rest on protocol-bound attacks and cross-domain evaluation; read the full paper before submission.
- [ ] [69] Orbisect: a search on 2026-10-11 found **no** publication or project with this name. Remove it and every sentence relying on it unless you can supply a source.
- [ ] The statements attributed to Sofy et al. [34] and Sa'ad et al. [38] against the actual texts.
- [ ] UNSW-IoTSAT and CuCD-ID **exist** (see Part 5) but are tabular. Decide what role each has; if you use one, cite it (CuCD-ID: Data in Brief doi 10.1016/j.dib.2026.112598, Mendeley doi 10.17632/7n2d42pm3n.3; UNSW-IoTSAT: Cyber Security and Applications vol. 4, 100133, 2026, verify with the publisher) and describe its tabular nature honestly. Adding them makes 71 references; that is fine (they are real), but they were not in your original 69.
- [ ] Novelty claim ("first benchmark of its kind"): repeat a literature search on the submission date; soften if anything close exists.
- [ ] Dataset terms of use and required citations for UNSW-NB15, CICIDS2017, NSL-KDD and NOS3's licence (NOSA).

### 10.2 Figures and format
- [ ] Draw Figure 1 (framework) and Figure 2 (packet structure and injection points; the table in Section VI is its content); Figures 3-5 are generated.
- [ ] Renumber references in order of appearance if the venue requires it (they currently follow your fixed [1]-[69] numbering).
- [ ] Lengths and template: the venue's page limit, anonymisation (double-blind: remove names, repository links and the acknowledgment).
- [ ] **AI-assistance disclosure:** many venues require disclosing AI assistance in writing or coding; check the venue's policy and follow it.
- [ ] Ask me for the IEEE LaTeX conversion once the text is final (the document also exports to Word, PDF and Markdown).

### 10.3 Reproducibility package
```bat
conda env export > environment_full.yml
pip freeze > requirements_frozen.txt
python --version
wmic cpu get name
systeminfo | findstr /C:"Total Physical Memory"
git status
```
- [ ] Keep: frozen config + commit hash, `environment_full.yml`, `requirements_frozen.txt`, `logs\`, `results_final\`, the NOS3 capture metadata, `nos3.pcap`/`attacks.csv`/`my_catalog.json` (the pcap is a derived artefact of an open simulator; check size limits, or publish the converted stream and labels).
- [ ] Do not publish raw terrestrial datasets (link to the providers); do not publish `results\cache`.
- [ ] Clean-room check: new environment from `environment_full.yml`, unzip the release, run the unit tests, then `python tools\run_seeds.py --config <final config> --seeds 1`. Numbers will not be bit-identical across machines; differences should be small and the conclusions unchanged.

### 10.4 Release the code
1. Choose a licence (none is set in the zip) and add `LICENSE`.
2. Create a GitHub repository, push the project (`.gitignore` already excludes cache, raw data and smoke results).
3. Tag a release; connect the repository to Zenodo to obtain a DOI (Zenodo's GitHub integration does this automatically for a release).
4. Put the repository URL and DOI in the Reproducibility Statement; for double-blind review use an anonymised mirror.

### 10.5 Submission pack
Manuscript (PDF/LaTeX), cover letter, data/code availability statement, ethics/dual-use statement (offline evaluation on
public data and your own simulator; attack code ships with its validator and refuses non-private targets), AI-disclosure
statement if required, reviewer suggestions, supplementary (config, tables CSV, figures).

---------------------------------------------------------------------------------------------------------------------
## Part 11. Troubleshooting
| Symptom | Fix |
|---|---|
| `'python' is not recognized` | Not in Anaconda Prompt or env not active: `conda activate ids`. |
| `ModuleNotFoundError: torch` / `art` | `pip install torch adversarial-robustness-toolbox` in the `ids` env. |
| `ModuleNotFoundError: No module named 'data'` | Run from the project root, or use `python -m data.preprocess ...`; update to the newest zip (fixed). |
| `DataNotFoundError ... file not found` | Wrong path/name: `python data\download_datasets.py --check`. |
| Download stops short (`ContentTooShortError`) | Use the updated `download_datasets.py` (retries) or `curl -L -o <file> <url>`. |
| `MemoryError` (CICIDS2017) | Lower `max_rows` in the config. |
| Very slow | Fewer epochs, `attacks.max_windows`, fewer models/datasets; measure with the pilot first. |
| `profile` error mentioning `tc_apid_list` | wire_format is cfs but `profile:` is missing: Part 5.7. |
| `No packets extracted` | Ports wrong (destination ports), wrong interface, or not UDP/IPv4: Part 5.3. |
| Many `parse_errors` | Wire format differs from your cFS build; send me the `tcpdump -X` of one command and one telemetry packet. |
| Checksum validates neither 0xFF nor 0x00 | Different cFE checksum rule or corrupted capture: tell me your cFE version. |
| A seed failed in `run_seeds.py` | Read the printed log tail, fix, run the same command again (finished seeds are kept). |
| Table 9 "not computable" | Missing CSVs / different `--out-dir`: run `make_tables.py --out-dir results_final` and check `dir results_final\tables`. |
| `PermissionError` on a CSV | Close it in Excel. |
| `UnicodeEncodeError` | `chcp 65001` then `set PYTHONUTF8=1`. |
| Table/figure has a SYNTHETIC banner | The input was synthetic; not a result. |

---------------------------------------------------------------------------------------------------------------------
## Appendix A. Command cheat sheet (Windows, project root, env `ids`)
```bat
python -m unittest discover -s tests                                        # tests
run_all.bat configs\smoke.yaml                                              # smoke
run_all.bat configs\pilot.yaml                                              # pilot
python data\download_datasets.py --check                                    # data present?
python tools\pcap_to_stream.py --pcap X --cmd-port C --tlm-port T --intervals attacks.csv --out data\raw\nos3\nos3
python tools\derive_profile.py --stream data\raw\nos3\nos3 --catalog tools\my_catalog.json
python tools\run_seeds.py --config configs\my_nos3.yaml --seeds 1 2 3 4 5 --final-dir results_final
python tools\merge_results.py --inputs results_s1 results_s2 --out results_final
python make_tables.py --config configs\my_nos3.yaml --out-dir results_final
```
## Appendix B. Where things are
`configs\` configs (smoke, pilot, default, per-dataset, nos3) - `data\` parser, schema, loaders, download helper -
`models\` six families + NumPy surrogate - `attacks\` standard, conformant, validator - `defenses\` AT, PAA -
`evaluation\` metrics, gap, tau, onboard - `experiments\` one script per phase - `tools\` NOS3 generator, pcap
converter, profile derivation, seed runner, merger - `results_*\tables|figures` outputs - `logs\` run logs.
## Appendix C. Design decisions the paper must state
Only attack-class samples are perturbed; accuracy is over the whole test set. Shared features (core 4 / extended 8),
TC = forward, TM = backward. LSTM reads the feature vector as a token sequence. Conformant attacker edits only TC
packets (padding, timing; APID in greedy), replay excluded, infeasible candidates count as failed attacks. AutoAttack =
APGD-CE only (ART). Random Forest attacked by transfer/greedy. PAA generated once, with the clean-augmentation
control. G uses PGD at the primary epsilon and pgd-ccsds (transfer-ccsds for non-differentiable models); tau is n/a when the
terrestrial gain is under 1 pp. Federated nodes with a single class skip perturbation. Latency is a host-CPU proxy.
