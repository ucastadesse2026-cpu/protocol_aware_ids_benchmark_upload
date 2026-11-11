# Data: what is included, what you must obtain

## Included (SYNTHETIC — for tests and the smoke configuration only)
* `samples/synthetic_ccsds_sample.bin` + `.labels.csv` — 24 windows x 32 packets of **synthetic** CCSDS-style
  packets (project mission profile, see `data/ccsds_parser.py`). Invented traffic statistics; no scientific meaning.
* `samples/SYNTHETIC_unsw_like_10k.csv` — 10,000 **synthetic** flow records with UNSW-NB15-style columns.
* Regenerate: `python data/make_samples.py`.
* **No real dataset is bundled.** The 10,000-row *real* preprocessed samples can be produced after you download the
  data: `python data/make_samples.py --real --config configs/default.yaml`.

## Public terrestrial datasets (you download them)
`python data/download_datasets.py --help` prints the official landing pages and expected paths. The script
contains no hard-coded URLs (hosting changes, licences may need acceptance). Place files at:
| Dataset | Expected path |
|---|---|
| UNSW-NB15 | `data/raw/unsw_nb15/UNSW_NB15_training-set.csv` |
| CICIDS2017 | `data/raw/cicids2017/*.csv` (the "MachineLearningCSV" files) |
| NSL-KDD | `data/raw/nsl_kdd/KDDTrain+.txt` |
### Where to get them (links checked 2026-10-10; none of the files were downloaded or integrity-checked by the authoring session)
| Dataset | Official page | How you actually get the files |
|---|---|---|
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset | The page's "HERE" link opens a UNSW SharePoint folder (browser download; the training/testing CSV partition is described on the page: 175,341 / 82,332 records). Free for academic use; cite the papers listed on the page. |
| CICIDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html | "Download this dataset" leads to a registration form at http://cicresearch.ca/CICDataset/CIC-IDS-2017/ ; get `MachineLearningCSV.zip` and extract all CSVs into `data/raw/cicids2017/`. An unofficial Hugging Face mirror lists the same zip: https://huggingface.co/datasets/bencorn/CICIDS2017 (use at your own risk; prefer the official source for a paper). |
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | The official page states the dataset is **no longer available**; its licence permits mirrors. Third-party GitHub mirror: https://github.com/Jehuty4949/NSL_KDD (`KDDTrain+.txt`). Check the result has 125,973 rows and 43 columns. |

Verify with `python data/download_datasets.py --check`. Only flow-level aggregates are used (see `feature_schema.py`).

## Satellite-domain data (checked 2026-10-11)
* **UNSW-IoTSAT** exists: repo https://github.com/Osama-Abdelhameed/UNSW-IoTSAT (MIT; data on a UNSW SharePoint folder linked
  from its README). It is a **tabular** dataset (404,798 records; 109 features + 3 CCSDS companion columns derived from
  attack labels) from a hybrid testbed (real sensors and CCSDS-framed BPSK radio, simulated orbit). It does not contain
  packet captures and has no packet/byte counts, so it cannot feed this benchmark's packet-level attacks or common features.
* **CuCD-ID** exists: Fayyaz, Yang, El-Khatib, *Data in Brief* 2026, doi 10.1016/j.dib.2026.112598; Mendeley Data
  doi 10.17632/7n2d42pm3n.3. NOS3/cFS records, **tabular** (25,000 x 31 raw; 22,465 x 23 noised; header fields and
  20-second window statistics), five scenarios, and **COSMOS scripts that reproduce its traffic** (not inspected by me).
* **NOS3/cFS simulator capture** (the route this repository supports): `docs/COMPLETE_GUIDE.md` Part 5.
* The benchmark expects satellite data as **a packet stream + per-packet labels**:
  * `capture.bin` - concatenated packets in the layout of `data/ccsds_parser.py` (formats "project" or "cfs")
  * `capture.labels.csv` - header `packet_index,label,attack_type[,arrival_time,...]`, one row per packet, `label` 1 = attack;
    the cFS format requires the `arrival_time` column (written by `tools/pcap_to_stream.py`).
  Set `data.satellite.source: real` and the paths in the config.

**Adapt the parser/mission profile to your data.** `ccsds_parser.py` implements the Space Packet primary header of
CCSDS 133.0-B-2 exactly; the "project" format's secondary header, opcode and CRC are synthetic profile choices, while
the "cfs" format follows the cFS command layout (function code + XOR checksum). If your source is a PCAP, use
`tools/pcap_to_stream.py`. Datasets that are not packet-level cannot be used for the conformant attacks.

### Real-data checklist (affects validity of the science)
1. Split satellite windows by capture session/time (chronological split is implemented).
2. Check that attack windows are valid under your profile (`tools/derive_profile.py`).
3. Record which missions/stacks/versions the captures come from; the limitations section must state it.
