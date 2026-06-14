# Netra RT Torch Inference — Rebrand Design Spec

**Date:** 2026-04-27  
**Approach:** Option A — CSS-variable + string reskin of `src/api/playground.html`  
**Reference design:** https://github.com/rifkybujana/netrart.com  

---

## Scope

Single file: `src/api/playground.html` (5,421 lines).  
All HTML structure and JavaScript logic remain untouched. Only CSS rules, CSS custom properties, font imports, and brand strings change.

---

## 1. Color Tokens

Replace the existing `:root` block and `[data-theme="dark"]` block.

| Variable | Light | Dark |
|----------|-------|------|
| `--bg` | `#f0f0f0` | `#121212` |
| `--surface` | `#ffffff` | `#1a1a1a` |
| `--surface2` | `#f8f8f8` | `#222222` |
| `--border` | `#e2e2e2` | `#2a2a2a` |
| `--border2` | `#eaeaea` | `#242424` |
| `--accent` | `#ff4800` | `#ff4800` |
| `--accent-hi` | `#ff6a33` | `#ff6a33` |
| `--accent-lo` | `#cc3a00` | `#cc3a00` |
| `--accent-bg` | `#fff3ee` | `#2a1200` |
| `--green` | `#3abc3f` | `#3abc3f` |
| `--green-bg` | `#f3fbf4` | `#0d2410` |
| `--green-text` | `#2e9632` | `#4ecb54` |
| `--red` | `#ff3131` | `#ff3131` |
| `--red-bg` | `#fff3f3` | `#2a0808` |
| `--red-text` | `#cc2727` | `#ff5555` |
| `--yellow` | `#ffa931` | `#ffa931` |
| `--yellow-bg` | `#fffaf3` | `#2a1f00` |
| `--yellow-text` | `#cc8727` | `#ffb74d` |
| `--text` | `#121212` | `#f0f0f0` |
| `--text-muted` | `#888888` | `#888888` |
| `--text-dim` | `#b5b5b5` | `#555555` |
| `--input-bg` | `#ffffff` | `#1a1a1a` |
| `--radius` | `0px` | `0px` |
| `--font` | `'Inter', system-ui, sans-serif` | same |
| `--serif` | `'Source Serif 4', Georgia, serif` | same |
| `--mono` | `'IBM Plex Mono', monospace` | same |

---

## 2. Typography

**Add to Google Fonts import:**
- `Source+Serif+4:ital,opsz,wght@0,8..60,200..900`
- `IBM+Plex+Mono:wght@400;500;600`
- Remove `Geist+Mono`

**Apply serif font (`var(--serif)`) to:**
- `.panel-title`
- `.card-title` (where used as a section heading)
- `.logo-name`

**Apply mono font (`var(--mono)`) to:**
- All existing `var(--mono)` usages — no selector changes needed, just the variable value changes.

---

## 3. Shape & Depth

- **All `border-radius` values → `0px`** across every component: buttons, cards, inputs, textareas, select, badges, tabs, dropzone, chat bubbles, stat cards, model cards, download tasks, GPU device blocks, response boxes, scrollbar thumbs.
- **Remove all `box-shadow`** from cards, inputs, buttons (flat design). Keep only functional shadows where they serve as focus rings (replace with outline-based approach using `#ff4800` at 20% opacity).
- **Focus ring:** `box-shadow: 0 0 0 2px rgba(255,72,0,0.2)` (replaces the current `var(--border2)` ring).
- **Scrollbar thumb:** `border-radius: 0px`, color `var(--border2)`.

---

## 4. Branding

| Location | Old | New |
|----------|-----|-----|
| `<title>` | `Kolosal Torch Inference` | `Netra RT — Torch Inference` |
| Favicon `<title>` text | `KI` | `NR` |
| Favicon `fill` | `#0066F5` | `#ff4800` |
| `.logo-mark` text | `KI` | `NR` |
| `.logo-mark` background | `var(--accent)` → blue | `#ff4800` (explicit) |
| `.logo-name` text | `Kolosal` | `Netra RT` |
| `.logo-sub` text | `Torch Inference` | `Torch Inference` (unchanged) |
| All `Kolosal` occurrences in body text | `Kolosal` | `Netra RT` |

---

## 5. Interactive States

| Component | Old | New |
|-----------|-----|-----|
| `.nav-item.active` bg | `--accent-bg` (blue-tinted) | `--accent-bg` (orange-tinted `#fff3ee`) |
| `.nav-item.active` color | `--accent` (blue) | `--accent` (orange) |
| `.tab.active` bg | `--accent-bg` | `--accent-bg` |
| `.tab.active` color | `--accent` | `--accent` |
| `.subtab.active` border | blue | `#ff4800` |
| `.btn-primary` hover gradient | `#0D0E0F → #FF3131 → #0066F5` | `#0D0E0F → #FF3131 → #ff4800` |
| `.msg.user .msg-body` bg | `--accent` (blue) | `--accent` (orange) |
| `.model-card:hover` border | `--accent` (blue) | `--accent` (orange) |
| `.dl-progress-fill` | `--accent` (blue) | `--accent` (orange) |
| `.dl-status-badge.downloading` | blue | orange: `#fff3ee` / `#cc3a00` |
| `.method.post` badge | blue | orange: `#fff3ee` / `#cc3a00` |
| `.ep-code` color | `--accent` | `--accent` |
| `.badge` bg/color | blue-tinted | orange-tinted |
| Hardcoded `#0066F5` / `#0052C4` / `#F0F6FE` | blue values | orange equivalents |

---

## 6. Select Dropdown Arrow

Update the SVG data-URI arrow in `select` and `[data-theme="dark"] select` — stroke color stays neutral (`#6A6F73` light / `#8b949e` dark), no color change needed.

---

## Out of Scope

- HTML structure
- JavaScript logic
- API endpoints
- Dark mode toggle mechanism
- Functional behavior of any panel
