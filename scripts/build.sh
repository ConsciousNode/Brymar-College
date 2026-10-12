#!/usr/bin/env bash
# scripts/build.sh
# MIT License — ConsciousNode SoftWorks
# Brymar College — build + package script
#
# Produces a distributable zip for Linux (primary).
# macOS / Windows packaging notes inline.
#
# Usage:
#   bash scripts/build.sh
#
# Prerequisites:
#   - Node.js ≥ 18
#   - @neutralinojs/neu installed globally: npm install -g @neutralinojs/neu
#   - sharp installed: npm install (handled by step 2)

set -e

echo ""
echo "  ■■■ BRYMAR COLLEGE — BUILD SCRIPT ■■■"
echo "  ConsciousNode SoftWorks"
echo ""

# ── Step 1: Generate icons ────────────────────────────────────────────────────
echo "[1/5] Generating icons..."
node scripts/gen-icons.js

# ── Step 2: Install / refresh Node dependencies ───────────────────────────────
echo "[2/5] Installing Node dependencies..."
npm install

# ── Step 3: Ensure Neutralino CLI is installed ────────────────────────────────
echo "[3/5] Checking Neutralino CLI..."
if ! command -v neu &> /dev/null; then
  echo "  [!] neu not found. Installing globally..."
  npm install -g @neutralinojs/neu
fi

# ── Step 4: Download Neutralino binaries if not present ──────────────────────
echo "[4/5] Updating Neutralino binaries..."
neu update

# ── Step 5: Build Neutralino distribution ─────────────────────────────────────
echo "[5/5] Building..."
neu build

# ── Package ───────────────────────────────────────────────────────────────────
VERSION=$(node -p "require('./package.json').version")
DIST_NAME="brymar-college-v${VERSION}"
DIST_DIR="dist/${DIST_NAME}"

mkdir -p "$DIST_DIR"

# Copy Neutralino build output
if [ -d "dist/brymar" ]; then
  cp -r dist/brymar/. "$DIST_DIR/"
else
  # Fallback: copy project files for CLI-only distribution
  echo "  [!] Neutralino binary not found. Packaging CLI-only distribution."
  cp -r gui/ "$DIST_DIR/gui/"
  cp index.js "$DIST_DIR/"
  cp package.json "$DIST_DIR/"
  cp -r src/ "$DIST_DIR/src/" 2>/dev/null || true
  cp -r training/ "$DIST_DIR/training/" 2>/dev/null || true
  cp -r persistence/ "$DIST_DIR/persistence/" 2>/dev/null || true
  cp -r tokenizer/ "$DIST_DIR/tokenizer/" 2>/dev/null || true
  cp -r multimodal/ "$DIST_DIR/multimodal/" 2>/dev/null || true
  cp -r model/ "$DIST_DIR/model/" 2>/dev/null || true
  cp -r extensions/ "$DIST_DIR/extensions/" 2>/dev/null || true
fi

# Always include these
cp scripts/install.sh "$DIST_DIR/"
cp README.md "$DIST_DIR/" 2>/dev/null || true
cp LICENSE  "$DIST_DIR/" 2>/dev/null || true

# ── Create zip ────────────────────────────────────────────────────────────────
cd dist
zip -r "${DIST_NAME}-linux.zip" "${DIST_NAME}/"
cd ..

echo ""
echo "  [OK] Build complete."
echo "  [OK] Output: dist/${DIST_NAME}-linux.zip"
echo ""
echo "  macOS notes:"
echo "    - Generate ICNS: cd gui/assets && iconutil -c icns icon.iconset"
echo "    - Create DMG: hdiutil create -volname 'Brymar College' -srcfolder dist/brymar -format UDZO dist/${DIST_NAME}-macos.dmg"
echo ""
echo "  Windows notes:"
echo "    - Neutralino produces brymar.exe in dist/brymar/"
echo "    - Use Inno Setup or NSIS for installer packaging"
echo ""
