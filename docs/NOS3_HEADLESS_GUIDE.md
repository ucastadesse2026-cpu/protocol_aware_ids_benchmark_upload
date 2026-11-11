# NOS3 on a command-line (headless) server: install -> run -> capture -> bring the data to your PC

Written for a Linux server you reach only by SSH (no desktop). `$` = a command typed on the **server**; `>` = a command
typed in **Anaconda Prompt on your Windows PC**. Every step ends with a **GATE**: do not continue until it holds; if it
does not, paste the output I ask for.

## What is verified and what is not
**Verified from NASA's NOS3 documentation and Makefile (read 2026-10-11):** NOS3 runs in Docker on Linux; install = Git,
Docker (+compose plugin), python3-pip/venv, `git clone`, `git submodule update --init --recursive`, `make prep`;
build with `make` (= `make all`: config, fsw, sim, gsw); `make launch` / `make stop` / `make stop-gsw` / `make clean` /
`make uninstall` / `make log` / `make debug`; the Makefile has a **headless target `ci-launch`** ("Headless Launch for
System Testing": runs `./scripts/ci_launch.sh`, then `./scripts/system_tests.sh`, then `./scripts/stop.sh`) with the
comment that it may break if COSMOS is not the configured ground system; ground system is chosen by `gsw` in
`cfg/nos3-mission.xml` (`cosmos` = COSMOS 4 desktop app, default; `openc3` = COSMOS 5 web; `yamcs`; `fprime`);
the spacecraft config `cfg/spacecraft/sc-mission-config.xml` has a switch for the 42 dynamics GUI;
`make config SC1_CFG=spacecraft/sc-minimal-config.xml` is the documented way to choose another spacecraft config;
cFS talks to the ground over UDP through the CI (command ingest) and TO (telemetry output) apps, **Debug** links active
by default, **Radio** TO link off until `TO_ENABLE_OUTPUT` is sent to target `CFS_RADIO`; all packets are CCSDS with the
secondary header; licence NOSA 1.3.
**NOT verified (GitHub blocked automated reading of the scripts):** what `ci_launch.sh` does exactly, container and
network names, the UDP port numbers, whether the system tests send enough different commands, whether a virtual
display fallback works. Those steps are written as "try, check, report" with exact checks.

Time and disk: unknown for your server. Docs: a VM disk of 32 GB may run NOS3, 64 GB+ recommended; 32 GB RAM may run,
64 GB+ recommended. Docker images and the build can take minutes to hours.

---------------------------------------------------------------------------------------------------------------------
## Part A. Before you start
**A1. Connect and keep sessions alive.** From Anaconda Prompt: `> ssh USER@SERVER_IP`. On the server:
```bash
$ sudo apt-get update && sudo apt-get install -y tmux unzip
$ tmux new -s nos3          # detach: Ctrl-b then d ; come back later: tmux attach -t nos3 ; new window: Ctrl-b c
```
Run everything long inside tmux so a dropped connection does not kill it.

**A2. Check the server.**
```bash
$ lsb_release -a ; nproc ; free -g ; df -h / ; git --version
```
GATE: Ubuntu (the docs' Docker steps are for Ubuntu), free disk well above 32 GB (aim 64 GB+), RAM as above. The docs list
**Git 2.47+**; if `git --version` is older and the clone or submodule step fails, install a newer Git
(`sudo add-apt-repository ppa:git-core/ppa && sudo apt-get update && sudo apt-get install -y git`).

---------------------------------------------------------------------------------------------------------------------
## Part B. Install Docker (from the NOS3 docs, Ubuntu)
```bash
$ sudo apt-get update
$ sudo apt-get install -y ca-certificates curl
$ sudo install -m 0755 -d /etc/apt/keyrings
$ sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
$ sudo chmod a+r /etc/apt/keyrings/docker.asc
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
$ sudo apt-get update
$ sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
$ sudo groupadd docker 2>/dev/null ; sudo usermod -aG docker $USER
```
Log out and back in (`exit`, then `ssh` again, then `tmux attach -t nos3` or a new tmux) so the group applies; the docs
use a reboot. Then:
```bash
$ docker run --rm hello-world
$ docker compose version
```
GATE: hello-world prints its greeting **without sudo**.

---------------------------------------------------------------------------------------------------------------------
## Part C. Get NOS3 and prepare it
```bash
$ cd ~ && git clone https://github.com/nasa/nos3.git
$ cd nos3 && git submodule update --init --recursive
$ sudo apt-get install -y python3-pip python3-venv python3-dev
$ python3 -m venv .venv && source .venv/bin/activate
$ make prep 2>&1 | tee ~/prep.log
```
(The docs describe the Python venv *after* `make prep` but say it is needed so the pip packages of `make prep` are
installed locally; creating it first is the safe order.) `make prep` downloads containers: expect many GB.
GATE: `make prep` ends without errors. If it fails (for example on the graphical Igniter tool, which `make prep` installs),
**paste the last 40 lines of `~/prep.log`** and stop.

Record the exact version for the paper (needed later):
```bash
$ git log -1 --format='%H %cd' ; git describe --tags --always ; git submodule status > ~/nos3_submodules.txt
$ docker images --digests > ~/nos3_images.txt ; df -h / | tail -1
```

---------------------------------------------------------------------------------------------------------------------
## Part D. Install the benchmark's capture tools on the server
On Windows (Anaconda Prompt, folder containing the zip):
```bat
> scp protocol_aware_ids_benchmark.zip USER@SERVER_IP:~/
```
On the server (use a SEPARATE environment from NOS3's `.venv`):
```bash
$ deactivate 2>/dev/null ; cd ~ && unzip -q protocol_aware_ids_benchmark.zip && cd protocol_aware_ids_benchmark
$ python3 -m venv ~/ids-tools && source ~/ids-tools/bin/activate && pip install numpy
$ python tools/pcap_flows.py --help | head -5 ; python tools/nos3_traffic.py --help | head -3
```
GATE: both `--help` commands print usage. (These four tools need only numpy.)

---------------------------------------------------------------------------------------------------------------------
## Part E. Configure for a headless run
```bash
$ cd ~/nos3 && source .venv/bin/activate
$ ls cfg cfg/spacecraft
$ grep -n "gsw" cfg/nos3-mission.xml
$ cp cfg/spacecraft/sc-mission-config.xml ~/sc-mission-config.xml.orig
$ grep -n -i "42\|gui" cfg/spacecraft/sc-mission-config.xml
```
1. **Ground system:** leave `gsw` at its default (`cosmos`, COSMOS 4). The Makefile warns the headless target may break
   with another one.
2. **42 GUI:** the docs say the spacecraft config enables or disables the 42 dynamics GUI. In the `grep` output find the
   42/GUI entry and set it to its "off" value (open with `nano cfg/spacecraft/sc-mission-config.xml`; if you are unsure
   which line, paste the grep output and I will tell you). A GUI window cannot open on a server without a display.
3. **Optional lighter system:** `make config SC1_CFG=spacecraft/sc-minimal-config.xml` (documented option). I have not
   seen what that minimal config contains: check afterwards that cFS command ingest and telemetry output still run (Part G).
GATE: you know which file you edited and have the backup `~/sc-mission-config.xml.orig`.

---------------------------------------------------------------------------------------------------------------------
## Part F. Build
```bash
$ cd ~/nos3 && source .venv/bin/activate
$ make 2>&1 | tee ~/build.log        # same as: make all  (config + fsw + sim + gsw)
```
Long step; keep it in tmux. GATE: it finishes without errors. Otherwise paste the last 40 lines of `~/build.log`.

---------------------------------------------------------------------------------------------------------------------
## Part G. Headless launch and health check
```bash
$ make stop                                     # harmless if nothing runs
$ ./scripts/ci_launch.sh 2>&1 | tee ~/launch.log
```
`ci_launch.sh` is what `make ci-launch` runs first (without the tests and the final stop). It may return to the prompt or
stay in the foreground: if it stays, open a second tmux window (Ctrl-b c) for the checks below.
```bash
$ docker ps --format 'table {{.Names}}\t{{.Status}}'
$ sleep 120 ; docker ps --format 'table {{.Names}}\t{{.Status}}' ; docker ps -a --filter status=exited --format '{{.Names}}'
```
GATE: several containers (flight software, simulators, COSMOS) show `Up` for 2+ minutes and none is `exited`.

**If it fails with DISPLAY / gnome-terminal / X errors** (the standard `make launch` opens terminal windows; older scripts used
`gnome-terminal`), try a virtual display, which is a generic workaround I have **not** tested with NOS3:
```bash
$ sudo apt-get install -y xvfb dbus-x11 gnome-terminal
$ dbus-run-session -- xvfb-run -a --server-args="-screen 0 1280x800x24" make launch 2>&1 | tee ~/launch2.log
```
If that fails too, **paste `~/launch.log` and `~/launch2.log` (last 40 lines each)**.

---------------------------------------------------------------------------------------------------------------------
## Part H. Find the network, the interface and the two UDP ports
```bash
$ docker network ls
$ NET=<the NOS3 network name from the list, e.g. one containing "nos3">
$ BR=br-$(docker network inspect -f '{{.Id}}' $NET | cut -c1-12) ; echo $BR ; ip -br link | grep "$BR"
$ docker inspect -f '{{.Name}} {{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' $(docker ps -q)
```
Write down the interface (`$BR`) and the IP of the **flight-software container**. Then capture one minute of UDP and let
the tool find the ports:
```bash
$ sudo timeout 60 tcpdump -i $BR -n -U -w ~/probe.pcap udp
$ cd ~/protocol_aware_ids_benchmark && source ~/ids-tools/bin/activate
$ python tools/pcap_flows.py --pcap ~/probe.pcap
```
(If the interface shows no packets, try `-i any`; with `any` the same datagram can appear twice and the converter removes
such duplicates.) GATE: the tool prints `suggested --cmd-port` **and** `suggested --tlm-port`, and the flow table makes
sense (TM flow = continuous, many packets; TC flow = few packets, may be empty if nothing is sending commands: then
run the discovery in Part I first and come back). If nothing parses, **paste the table and
`sudo timeout 10 tcpdump -i $BR -n -c 5 -X udp`**: the wire format then differs from the cFS layout I assumed.

Telemetry note: only the Debug links are active by default; the Radio telemetry link needs `TO_ENABLE_OUTPUT` to target
`CFS_RADIO`. The Debug channel is enough for this study; say in the paper which channel was captured.

---------------------------------------------------------------------------------------------------------------------
## Part I. Discovery run: harvest real commands (no guessing of command IDs)
Goal: obtain real command packets (APID, function code, payload) from NOS3's own system tests, so the catalog contains
commands that exist in your build.
```bash
$ cd ~/nos3 && source .venv/bin/activate && make stop
$ sudo tcpdump -i $BR -n -U -w ~/discovery.pcap udp &        # note the job; stop it later with: sudo pkill tcpdump
$ make ci-launch 2>&1 | tee ~/ci.log                          # launch + system tests + stop (documented headless target)
$ sudo pkill tcpdump
$ cd ~/protocol_aware_ids_benchmark && source ~/ids-tools/bin/activate
$ python tools/pcap_flows.py --pcap ~/discovery.pcap
$ python tools/pcap_to_stream.py --pcap ~/discovery.pcap --cmd-port CMD --tlm-port TLM --out ~/discovery
$ python tools/catalog_from_stream.py --stream ~/discovery --out tools/my_catalog.json
```
(Replace CMD and TLM with the ports the tool suggested.) `catalog_from_stream.py` lists every distinct (APID, function
code) it saw with its most common payload and **suggests** a class (frequent = benign, rare = sensitive).
**You must review the catalog before using it** (`nano tools/my_catalog.json`):
* The tool cannot know what a command does. Look each one up (COSMOS command definitions in the repo: find them with
  `grep -rIl "<APID in hex>" ~/nos3/gsw 2>/dev/null | head`; paths differ by version) and **delete anything that could
  reset, delete files, power devices off or otherwise damage the run**.
* Keep at least one `benign` and one `sensitive` entry. If the system tests sent too little variety, add entries by hand
  from the COSMOS definitions.
GATE: `tools/my_catalog.json` exists, every entry reviewed, >= 1 benign and >= 1 sensitive. If `ci.log` shows the system
tests failed or sent no commands, paste its last 40 lines.

---------------------------------------------------------------------------------------------------------------------
## Part J. The real capture session
You need three tmux windows: (1) NOS3, (2) tcpdump, (3) generator.
```bash
# window 1
$ cd ~/nos3 && source .venv/bin/activate && make stop && ./scripts/ci_launch.sh 2>&1 | tee ~/launch_real.log
$ docker ps --format 'table {{.Names}}\t{{.Status}}'        # same GATE as Part G
# window 2  (capture ONE interface to avoid duplicates)
$ mkdir -p ~/capture && sudo tcpdump -i $BR -n -U -w ~/capture/nos3.pcap udp
# let it run ~5 minutes with NO generator: this is the clean benign baseline
# window 3
$ cd ~/protocol_aware_ids_benchmark && source ~/ids-tools/bin/activate
$ python tools/nos3_traffic.py --host FSW_IP --port CMD --catalog tools/my_catalog.json --mode session \
      --cycles 20 --benign-s 60 --attack-s 20 --intervals-out ~/capture/attacks.csv --i-own-this-simulator
```
(`FSW_IP` = the flight-software container IP from Part H; Docker bridge addresses are private, so the generator accepts
them.) Watch for trouble while it runs (window 1 or another): `docker ps`, `docker logs --tail 20 FSW_CONTAINER`; if the
flight software crashes or stops answering during a flood, stop, `make stop`, lower the flood rate (`--rates
benign=1,flood=30,injection=2,recon=5`) and restart.
When the generator finishes: Ctrl-C the tcpdump window, then `make stop` in window 1, then `docker ps` (nothing left).
GATE: `ls -lh ~/capture` shows `nos3.pcap` and `attacks.csv` (3 lines or more); `python tools/pcap_flows.py --pcap
~/capture/nos3.pcap` shows the same two ports as before.

How much data? The number of windows is what matters (32 packets each). Convert (Part K) and read the split table of
`derive_profile.py`; if a chronological third lacks attack windows or the counts are small, run more `--cycles`
(append to a new capture, then merge by recapturing once with more cycles; do not splice files by hand).

---------------------------------------------------------------------------------------------------------------------
## Part K. Convert on the server, then fetch the results
```bash
$ cd ~/protocol_aware_ids_benchmark && source ~/ids-tools/bin/activate
$ python tools/pcap_to_stream.py --pcap ~/capture/nos3.pcap --cmd-port CMD --tlm-port TLM \
      --intervals ~/capture/attacks.csv --out ~/capture/nos3
```
GATE: it prints counts; `parse_errors` is near 0, `attack_packets` > 0, `duplicates` small. Then, on Windows:
```bat
> mkdir data\raw\nos3
> scp USER@SERVER_IP:~/capture/nos3.bin data\raw\nos3\
> scp USER@SERVER_IP:~/capture/nos3.labels.csv data\raw\nos3\
> scp USER@SERVER_IP:~/capture/attacks.csv USER@SERVER_IP:~/protocol_aware_ids_benchmark/tools/my_catalog.json .
> scp USER@SERVER_IP:~/capture/nos3.pcap USER@SERVER_IP:~/nos3_submodules.txt USER@SERVER_IP:~/nos3_images.txt .
```
(Keep the pcap, the version files and the catalog for the paper's reproducibility package.)

---------------------------------------------------------------------------------------------------------------------
## Part L. Continue on Windows
Follow `docs/COMPLETE_GUIDE.md` Part 5.7 (derive the profile), Part 6 (freeze the config), Part 7 (full runs) and on.
```bat
> python tools\derive_profile.py --stream data\raw\nos3\nos3 --catalog tools\my_catalog.json
```
Gate there: hundreds of windows and attack windows in train, val and test.

---------------------------------------------------------------------------------------------------------------------
## Part M. Shut down and clean up (server)
```bash
$ cd ~/nos3 && source .venv/bin/activate
$ make stop ; make stop-gsw ; docker ps            # nothing should be running
$ make clean                                        # removes build files (rebuild with make)
$ make uninstall                                    # only when you no longer need NOS3 (undoes make prep)
```

---------------------------------------------------------------------------------------------------------------------
## Part N. What the paper must say about this data
* Satellite-domain data = **NOS3 (NOSA 1.3 licence) simulator traffic** captured on the cFS **Debug** UDP links of one
  spacecraft, with **author-scripted** attack classes (flood, injection, recon) and ground-truth labels from the
  generator's interval log; benign traffic = NOS3's own telemetry plus scripted routine commands.
* Report: NOS3 commit hash, submodules, Docker image digests, spacecraft config (and any change such as the 42 GUI off),
  ground system, capture interface and command, catalog (with how it was derived and reviewed), rates, durations, cycles.
* Suggested citation for NOS3 (verify details): "NASA Operational Simulator for Small Satellites (NOS3): the STF-1 CubeSat
  case study", arXiv:1901.07583. If you reproduce CuCD-ID scenarios, cite it too (Data in Brief, doi 10.1016/j.dib.2026.112598).
* Say plainly: simulated, scripted, regular benign traffic, one spacecraft, TC-only attacker; results do not transfer
  automatically to flight systems.

---------------------------------------------------------------------------------------------------------------------
## Part O. Troubleshooting
| Symptom | What to do |
|---|---|
| `permission denied ... docker.sock` | Log out/in so the docker group applies, or `newgrp docker`. |
| `make prep` / `make` fails | Paste the last 40 lines of `~/prep.log` / `~/build.log`; check `df -h /` for free disk. |
| Launch complains about DISPLAY / gnome-terminal | Part G virtual-display fallback; else send the logs. |
| Containers `exited` right after launch | `docker logs --tail 40 CONTAINER`; send me the output. |
| No UDP packets on `$BR` | Try `-i any`; check `docker ps`; the TO link may not be sending: wait 2 minutes after launch. |
| `pcap_flows.py` finds no TC flow | Nothing sent commands yet: do Part I (system tests) or start the generator. |
| Flows found but nothing parses | Wire format differs: send me the `-X` hex dump of a few packets. |
| Generator: "refusing to send" | The host is not private/loopback: use the container IP shown by `docker inspect`. |
| cFS stops responding during the flood | Lower `--rates flood=...`, restart NOS3 (`make stop`, relaunch). |
| Huge pcap | Capture one interface, not `any`; shorten `--cycles`; check `du -h ~/capture`. |
| `make stop` leaves containers | `docker ps` then `docker stop NAME`; `make stop-gsw`. |

## Part P. What to paste back to me, by gate
A2 (resources) - C (`prep.log` tail if it failed) - F (`build.log` tail if it failed) - G (`docker ps` output, launch logs on
failure) - H (flow table and suggested ports) - I (catalog review questions) - K (conversion counts) - L (derive_profile output).
