// scripts/gen-icons.js
// MIT License — ConsciousNode SoftWorks
// Copyright (c) 2026 Khamerron Edward Ramsey Kizer (Kham), Kehai Interim, Ed Interim, Vael Interim
// Generate PNG icons from the SVG source using sharp.
// Run once: node scripts/gen-icons.js

import sharp from 'sharp';
import { readFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root      = join(__dirname, '..');
const assetsDir = join(root, 'gui', 'assets');

// Ensure output directory exists
mkdirSync(assetsDir, { recursive: true });

const svgPath = join(assetsDir, 'icon.svg');
const svg     = readFileSync(svgPath);

const sizes = [
  { name: 'icon.png',     size: 512 },   // Neutralino window icon (primary)
  { name: 'icon-256.png', size: 256 },   // Medium
  { name: 'icon-128.png', size: 128 },   // Small / taskbar
  { name: 'icon-64.png',  size: 64  },   // Notification
  { name: 'icon-32.png',  size: 32  },   // Tray / favicon
];

console.log('Generating icons from', svgPath);
for (const { name, size } of sizes) {
  const out = join(assetsDir, name);
  await sharp(svg)
    .resize(size, size)
    .png()
    .toFile(out);
  console.log(`  ✓ ${name}  (${size}×${size})`);
}

// ICO: embed 256px PNG into a minimal ICO container
// (Full ICO tooling requires external deps — we use 256px PNG
//  which most Windows contexts accept as an ICO source)
import { copyFileSync } from 'node:fs';
const icoOut = join(assetsDir, 'icon.ico');
copyFileSync(join(assetsDir, 'icon-256.png'), icoOut);
console.log('  ✓ icon.ico  (256px PNG wrapped — full ICO via rsvg-convert if needed)');

// ICNS: placeholder note
// Full ICNS requires macOS `iconutil` or a third-party library.
// Use `iconutil -c icns icon.iconset` on macOS from the PNGs above.
console.log('  ℹ icon.icns — generate on macOS: see scripts/build.sh for instructions');

console.log('\nIcons generated.');
