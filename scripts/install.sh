#!/usr/bin/env bash
# scripts/install.sh
# MIT License — ConsciousNode SoftWorks
# Brymar College — end-user installer
#
# Run from the extracted distribution directory.
# Creates corpus/ and checkpoints/ directories, installs desktop icon.
#
# Usage:
#   bash install.sh

set -e

INSTALL_DIR="$HOME/.local/share/brymar-college"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/512x512/apps"

echo ""
echo "  ■■■  BRYMAR COLLEGE  ■■■"
echo "  ConsciousNode SoftWorks"
echo ""
echo "  Installing to: $INSTALL_DIR"
echo ""

# ── Step 1: Copy distribution files ──────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
cp -r . "$INSTALL_DIR/"

# Make binary executable
[ -f "$INSTALL_DIR/brymar" ] && chmod +x "$INSTALL_DIR/brymar"

# ── Step 2: Create working directories ───────────────────────────────────────
mkdir -p "$INSTALL_DIR/corpus"
mkdir -p "$INSTALL_DIR/checkpoints"
mkdir -p "$INSTALL_DIR/exports"
mkdir -p "$INSTALL_DIR/logs"

echo "  Created: corpus/       — place training text files here"
echo "  Created: checkpoints/  — model checkpoints saved here"
echo "  Created: exports/      — exported LoRA adapters"
echo "  Created: logs/         — training logs"

# ── Step 3: Symlink CLI binary ────────────────────────────────────────────────
mkdir -p "$BIN_DIR"

if [ -f "$INSTALL_DIR/brymar" ]; then
  ln -sf "$INSTALL_DIR/brymar" "$BIN_DIR/brymar"
  echo "  CLI binary symlinked: $BIN_DIR/brymar"
else
  # CLI-only distribution (no Neutralino binary)
  cat > "$BIN_DIR/brymar" << EOF
#!/usr/bin/env bash
node "$INSTALL_DIR/index.js" "\$@"
EOF
  chmod +x "$BIN_DIR/brymar"
  echo "  CLI wrapper created: $BIN_DIR/brymar"
fi

# ── Step 4: Desktop icon ──────────────────────────────────────────────────────
mkdir -p "$DESKTOP_DIR" "$ICON_DIR"

ICON_SRC="$INSTALL_DIR/gui/assets/icon.png"

if [ -f "$ICON_SRC" ]; then
  cp "$ICON_SRC" "$ICON_DIR/brymar-college.png"

  cat > "$DESKTOP_DIR/brymar-college.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Brymar College
GenericName=LoRA Fine-Tuning Studio
Comment=ConsciousNode LoRA fine-tuning studio for RWKV-v7
Exec=$INSTALL_DIR/brymar
Icon=brymar-college
Terminal=false
Categories=Science;Education;Development;
Keywords=AI;machine learning;RWKV;LoRA;fine-tuning;
StartupNotify=true
StartupWMClass=brymar
EOF

  chmod +x "$DESKTOP_DIR/brymar-college.desktop"

  # Refresh icon cache
  update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
  gtk-update-icon-cache "$HOME/.local/share/icons/hicolor/" 2>/dev/null || true

  echo "  Desktop entry created: $DESKTOP_DIR/brymar-college.desktop"
  echo "  Icon installed: $ICON_DIR/brymar-college.png"
else
  echo "  [!] Icon not found at $ICON_SRC — desktop entry skipped"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "  ─────────────────────────────────────────"
echo "  [OK] Installation complete."
echo ""
echo "  To train:"
echo "    Place your corpus files in: $INSTALL_DIR/corpus/"
echo "    Then run:                   brymar train"
echo "    Or launch from your app menu: Brymar College"
echo ""
echo "  If $BIN_DIR is not in your PATH, add this to ~/.bashrc:"
echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  ─────────────────────────────────────────"
echo ""
