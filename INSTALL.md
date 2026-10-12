# Brymar College — Installation & Transfer Guide

**ConsciousNode SoftWorks — May 2026**

This document covers:

1. [System Requirements](#1-system-requirements)
2. [Installing on Linux](#2-installing-on-linux)
3. [Installing on Windows](#3-installing-on-windows)
4. [Transferring to Another Machine](#4-transferring-to-another-machine)
5. [Verifying the Installation](#5-verifying-the-installation)
6. [Uninstalling](#6-uninstalling)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. System Requirements

### Minimum (tested target)
| Component | Minimum |
|---|---|
| CPU | Intel Core i7 Sandy Bridge (2011) or equivalent |
| RAM | 8 GB physical + zRAM swap recommended |
| Storage | 6 GB free (3 GB model + project + corpus) |
| OS | Linux (Ubuntu 20.04+), Windows 10+, macOS 11+ |
| Node.js | **18.0 or later** (LTS recommended) |
| npm | 9.0 or later (bundled with Node.js) |
| Internet | Required once (model download ~3 GB), offline after |

### Recommended
| Component | Recommended |
|---|---|
| RAM | 16 GB |
| Storage | SSD, 20 GB free |
| OS | Ubuntu 22.04 LTS or Windows 11 |
| Node.js | 22 LTS |

> **No GPU required.** All training and inference runs on CPU.  
> The GUI uses the system's native WebView — no Chromium, no Electron, no extra 120 MB runtime.

---

## 2. Installing on Linux

### 2a. Install Node.js

**Option A — NodeSource (recommended, stays up to date):**
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # should print v22.x.x or similar
```

**Option B — NVM (lets you manage multiple versions):**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 22
nvm use 22
```

**Option C — Snap:**
```bash
sudo snap install node --classic --channel=22
```

### 2b. Get the Brymar College project

**From a zip transfer (most common — see Section 4):**
```bash
unzip brymar-college-v3.3.4.zip -d ~/brymar-college
cd ~/brymar-college
```

**From a folder copy (USB drive or network share):**
```bash
cp -r /media/usb/brymar-college ~/brymar-college
cd ~/brymar-college
```

### 2c. Install Node dependencies

```bash
npm install
```

This installs: `ws`, `chalk`, `commander`, `cli-progress`, `@huggingface/hub`, and optionally `sharp`.  
No internet is needed after this step for training — only for first model download.

### 2d. Download Neutralino runtime binaries

The GUI requires Neutralino v6 binaries (already included if you received a full transfer zip).  
If they are missing (fresh clone):

```bash
# Install Neutralino CLI globally (one-time)
npm install -g @neutralinojs/neu

# Download Neutralino runtime binaries into ./bin/
neu update
```

### 2e. Generate icons (optional — for GUI)

```bash
node scripts/gen-icons.js
```

This generates `gui/assets/icon.png` and sizes. Requires `sharp` (installed in step 2c).

### 2f. Run the end-user installer (optional — adds desktop icon + CLI shortcut)

```bash
bash scripts/install.sh
```

This:
- Copies the project to `~/.local/share/brymar-college/`
- Creates `~/brymar-college/corpus/`, `checkpoints/`, `exports/`, `logs/`
- Symlinks `brymar` into `~/.local/bin/brymar`
- Installs a `.desktop` entry (appears in GNOME / KDE app launcher)

To make the CLI available in your shell immediately:
```bash
export PATH="$HOME/.local/bin:$PATH"
# Or add that line to ~/.bashrc for permanent effect:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2g. Launch

**GUI:**
```bash
cd ~/brymar-college
npm run gui
# or: neu run
```

**CLI (always works independently):**
```bash
node index.js train --corpus ./corpus --steps 2000 --rank 4
```

**After running `install.sh`:**
```bash
brymar train          # CLI via installed shortcut
# or double-click "Brymar College" in your app launcher
```

---

## 3. Installing on Windows

### 3a. Install Node.js

1. Go to [nodejs.org](https://nodejs.org) and download the **LTS** installer (`.msi`).
2. Run the installer. Accept defaults. Make sure "Add to PATH" is checked.
3. Open **PowerShell** and verify:
   ```powershell
   node --version
   npm --version
   ```

> If you see `'node' is not recognized`, restart PowerShell or open a new terminal window after install.

### 3b. Get the Brymar College project

**From a zip:**
```powershell
# Right-click the zip in Explorer → "Extract All"
# Or in PowerShell:
Expand-Archive -Path "brymar-college-v3.3.3.zip" -DestinationPath "C:\Users\YourName\brymar-college"
cd "C:\Users\YourName\brymar-college"
```

**From a USB drive or folder copy:**
```powershell
Copy-Item -Recurse "E:\brymar-college" "C:\Users\YourName\brymar-college"
cd "C:\Users\YourName\brymar-college"
```

### 3c. Install Node dependencies

```powershell
npm install
```

### 3d. Download Neutralino runtime binaries (if missing)

```powershell
# Install Neutralino CLI globally (one-time)
npm install -g @neutralinojs/neu

# Download runtime binaries
neu update
```

### 3e. Generate icons

```powershell
node scripts\gen-icons.js
```

### 3f. Launch

**GUI:**
```powershell
npm run gui
```

**CLI:**
```powershell
node index.js train --corpus .\corpus --steps 2000 --rank 4
```

### 3g. Optional: Add CLI to PATH

To run `brymar` from anywhere in PowerShell:

1. Press `Win + R` → type `sysdm.cpl` → OK
2. Advanced tab → Environment Variables
3. Under "User variables", select `Path` → Edit → New
4. Add the full path to your `brymar-college` folder (e.g. `C:\Users\YourName\brymar-college`)
5. Click OK everywhere, restart PowerShell

Alternatively, create a simple batch file wrapper:

```powershell
# Create brymar.cmd in a folder that's already on your PATH
# Example: C:\Users\YourName\AppData\Local\Microsoft\WindowsApps\
$content = "@echo off`nnode `"C:\Users\YourName\brymar-college\index.js`" %*"
Set-Content "C:\Users\YourName\AppData\Local\Microsoft\WindowsApps\brymar.cmd" $content
```

Then:
```powershell
brymar train
```

### 3h. Windows Desktop Shortcut (GUI)

1. Right-click your Desktop → New → Shortcut
2. Target: `C:\Users\YourName\brymar-college\bin\neutralino-win_x64.exe`
3. Name: `Brymar College`
4. Right-click the shortcut → Properties → Change Icon → browse to `gui\assets\icon.ico`

---

## 4. Transferring to Another Machine

### Method A — Zip transfer (recommended, works offline)

This is the cleanest method. It bundles everything including `node_modules` and the Neutralino
binaries so the destination machine needs **only Node.js**, nothing else.

**On the source machine:**

```bash
# Linux / macOS
cd /path/to/parent-folder

zip -r brymar-college-transfer.zip brymar-college/ \
  --exclude "brymar-college/.git/*" \
  --exclude "brymar-college/corpus/*" \
  --exclude "brymar-college/checkpoints/*" \
  --exclude "brymar-college/exports/*" \
  --exclude "brymar-college/logs/*"
```

```powershell
# Windows PowerShell
Compress-Archive -Path "C:\Users\YourName\brymar-college" `
  -DestinationPath "C:\Users\YourName\brymar-college-transfer.zip"
```

> **Include `node_modules/` and `bin/`** in the transfer zip.  
> This means the destination machine does **not** need internet access.  
> Zip size with dependencies: approximately 50–80 MB (excl. corpus and checkpoints).

**Transfer the zip:**
- USB drive
- Local network share (`\\server\share` on Windows, `scp` or `rsync` on Linux)
- Syncthing (LAN sync, no cloud)
- Any physical media

**On the destination machine:**
```bash
# Linux
unzip brymar-college-transfer.zip -d ~/
cd ~/brymar-college
node --version      # verify Node.js is installed
node index.js train --help   # verify CLI works
npm run gui                  # verify GUI works
```

```powershell
# Windows
Expand-Archive -Path "brymar-college-transfer.zip" -DestinationPath "$HOME\"
cd "$HOME\brymar-college"
node index.js train --help
```

---

### Method B — Minimal transfer (internet required on destination)

If the destination machine has reliable internet, you can transfer just the project source
(no `node_modules`, no `bin/`) and re-install on arrival. Much smaller transfer (~2 MB).

**On source:**
```bash
# Exclude generated and dependency directories
zip -r brymar-college-source.zip brymar-college/ \
  --exclude "brymar-college/node_modules/*" \
  --exclude "brymar-college/bin/*" \
  --exclude "brymar-college/.git/*" \
  --exclude "brymar-college/corpus/*" \
  --exclude "brymar-college/checkpoints/*" \
  --exclude "brymar-college/exports/*"
```

**On destination (requires internet):**
```bash
unzip brymar-college-source.zip -d ~/
cd ~/brymar-college
npm install                    # re-downloads Node dependencies
npm install -g @neutralinojs/neu
neu update                     # re-downloads Neutralino binaries
node scripts/gen-icons.js      # re-generates icons
npm run gui
```

---

### Method C — Transferring trained checkpoints only

If you only want to move a trained LoRA adapter to a different machine:

```bash
# Copy just the checkpoint file
scp ./checkpoints/checkpoint-002000.json user@remote:~/brymar-college/checkpoints/

# Then on the remote machine, resume training or run inference:
node index.js train --resume ./checkpoints/checkpoint-002000.json --steps 500
```

Checkpoint files are small (a few MB per checkpoint — they contain only LoRA A/B matrices).

---

### What to include in any transfer

| Item | Include? | Notes |
|---|---|---|
| `index.js` | ✅ Always | CLI entry point |
| `src/` | ✅ Always | Config + terminal |
| `model/` | ✅ Always | RWKV-v7 model stack |
| `training/` | ✅ Always | Training loop |
| `tokenizer/` | ✅ Always | BPE tokenizer |
| `multimodal/` | ✅ Always | ElasticTok + SpikeVox |
| `persistence/` | ✅ Always | Checkpoint I/O |
| `hardware/` | ✅ Always | OpenCL + zRAM |
| `gui/` | ✅ Always | Desktop GUI |
| `extensions/` | ✅ Always | Neutralino bridge |
| `scripts/` | ✅ Always | Build + install |
| `neutralino.config.json` | ✅ Always | GUI config |
| `package.json` | ✅ Always | Dependencies manifest |
| `node_modules/` | ⚠️ Optional | Include for offline transfer; exclude to save space |
| `bin/` | ⚠️ Optional | Include for offline GUI; exclude to save space |
| `corpus/` | 🔒 Your choice | Your training data — personal |
| `checkpoints/` | 🔒 Your choice | Your trained adapters |
| `.git/` | ❌ Skip | Version control history, not needed |
| `Source Files for reference. READ ONLY/` | ❌ Skip | Large reference files, not part of runtime |

---

## 5. Verifying the Installation

Run these checks after any install or transfer:

```bash
# 1. Node.js version (must be ≥ 18)
node --version

# 2. Dependencies installed
node -e "import('chalk').then(m => console.log('chalk ok')).catch(() => console.log('MISSING: run npm install'))"

# 3. CLI works
node index.js --help

# 4. Phase 4 smoke test (tokenizer + multimodal)
node --input-type=module --eval "
import { NanoTokenizer } from './tokenizer/bpe.js';
const t = new NanoTokenizer();
await t.train('hello world hello', 260);
console.log('BPE ok. Vocab:', t.size);
console.log('Encode:', t.encode('hello'));
console.log('Decode:', t.decode(t.encode('hello')));
"

# 5. GUI binaries present
node -e "
const { existsSync } = require('fs');
const ok = existsSync('./bin/neutralino-linux_x64') || existsSync('./bin/neutralino-win_x64.exe');
console.log(ok ? 'Neutralino binaries: ok' : 'Neutralino binaries: MISSING — run: neu update');
"
```

Expected output on a healthy install:
```
v22.x.x
chalk ok
Usage: brymar [command] [options]
BPE ok. Vocab: 260
Encode: [ 259, ... ]
Decode: hello
Neutralino binaries: ok
```

---

## 6. Uninstalling

### Linux (if installed via `install.sh`)
```bash
rm -rf ~/.local/share/brymar-college
rm -f  ~/.local/bin/brymar
rm -f  ~/.local/share/applications/brymar-college.desktop
rm -f  ~/.local/share/icons/hicolor/512x512/apps/brymar-college.png
update-desktop-database ~/.local/share/applications/ 2>/dev/null || true
```

### Windows
```powershell
Remove-Item -Recurse -Force "C:\Users\YourName\brymar-college"
# Remove shortcut from Desktop if you created one
# Remove PATH entry via System Properties if you added one
```

---

## 7. Troubleshooting

### `node: command not found` (Linux)
Node.js is not installed or not in PATH.  
→ Follow Section 2a. If using NVM, run `source ~/.bashrc` and try again.

### `npm install` fails with `EACCES permission denied`
You're trying to install into a system directory.  
→ Never use `sudo npm install`. Use NVM instead (Section 2a Option B).

### `neu: command not found`
Neutralino CLI is not installed globally.  
→ `npm install -g @neutralinojs/neu`  
→ On Linux: ensure `~/.npm-global/bin` is in PATH, or use: `npx neu run`

### GUI opens a blank window
The Neutralino client library `gui/neutralino.js` is missing.  
→ `neu update` (re-downloads both the binary and the JS client)

### `sharp` install fails on Windows
Sharp uses native binaries. If the pre-built binary doesn't match your Node.js version:  
→ `npm install --ignore-scripts sharp` (icons won't generate; all other features work)  
→ Icons can be provided manually: any 512×512 PNG named `gui/assets/icon.png`

### Training crashes immediately with `Cannot find module`
Dependencies not installed.  
→ `npm install`

### `out of memory` during training
The model exceeds available RAM.  
→ Use the 1.5B model (not 2.9B) for systems with less than 16 GB RAM  
→ On Linux: enable zRAM (`sudo modprobe zram && sudo zramctl`)  
→ Reduce `--seqLen` in `src/config.js`

### BPE training hangs for a long time
The Worker Thread is processing a large corpus. This is normal for > 5 MB corpora.  
Progress is logged every 20%. For very large corpora, use `--byteLevel` mode
(no BPE training — raw UTF-8 bytes as tokens):
```bash
node index.js train --byteLevel
```

### `ws` module not found (extension error)
```bash
npm install ws
```

---

*MIT License — Copyright (c) 2026 ConsciousNode SoftWorks*  
*Authors: Khamerron Edward Ramsey Kizer, Kehai Interim, Ed Interim, Vael Interim*
