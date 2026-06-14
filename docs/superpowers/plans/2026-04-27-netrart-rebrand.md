# Netra RT Rebrand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Visually rebrand `src/api/playground.html` from Kolosal (blue, rounded) to Netra RT (orange, square, flat) using the netrart.com design language, preserving all HTML structure and JavaScript logic.

**Architecture:** CSS-variable-first reskin — replace token blocks, remove shadows, zero non-circular border-radii, swap hardcoded blue hex/rgba values, update brand strings. No structural HTML or JS changes.

**Tech Stack:** Single embedded HTML file (5,421 lines), vanilla CSS, vanilla JS. Build: `cargo build --release` embeds the file via `include_str!`.

---

## Files

- **Modify:** `src/api/playground.html` — only file touched

---

## Task 1: Update font imports

**Files:**
- Modify: `src/api/playground.html:10`

- [ ] **Step 1: Replace the Google Fonts `<link>` on line 10**

Use Edit tool. Replace:
```
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap" rel="stylesheet" />
```
With:
```
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=Source+Serif+4:ital,opsz,wght@0,8..60,200..900;1,8..60,200..900&display=swap" rel="stylesheet" />
```

- [ ] **Step 2: Verify**
```bash
grep -n "Geist\|IBM Plex\|Source Serif" src/api/playground.html | head -5
```
Expected: one line showing `IBM+Plex+Mono` and `Source+Serif+4`, zero lines showing `Geist`.

- [ ] **Step 3: Commit**
```bash
git add src/api/playground.html
git commit -m "style: swap fonts — IBM Plex Mono + Source Serif 4 (drop Geist Mono)"
```

---

## Task 2: Replace `:root` CSS token block

**Files:**
- Modify: `src/api/playground.html:15-41`

- [ ] **Step 1: Replace the `:root` block**

Use Edit tool. Replace:
```css
  :root {
    --bg:        #F8F9F9;
    --surface:   #FFFFFF;
    --surface2:  #FFFFFF;
    --border:    #DDE1E3;
    --border2:   #E4E7E9;
    --accent:    #0066F5;
    --accent-hi: #3C8AF7;
    --accent-lo: #003D93;
    --accent-bg: #F0F6FE;
    --green:     #3ABC3F;
    --green-bg:  #F3FBF4;
    --green-text:#2E9632;
    --red:       #FF3131;
    --red-bg:    #FFF3F3;
    --red-text:  #CC2727;
    --yellow:    #FFA931;
    --yellow-bg: #FFFAF3;
    --yellow-text:#CC8727;
    --text:      #0D0E0F;
    --text-muted:#6A6F73;
    --text-dim:  #9C9FA1;
    --input-bg:  #FFFFFF;
    --radius:    10px;
    --font:      'Inter', system-ui, sans-serif;
    --mono:      'Geist Mono', monospace;
  }
```
With:
```css
  :root {
    --bg:        #f0f0f0;
    --surface:   #ffffff;
    --surface2:  #f8f8f8;
    --border:    #e2e2e2;
    --border2:   #eaeaea;
    --accent:    #ff4800;
    --accent-hi: #ff6a33;
    --accent-lo: #cc3a00;
    --accent-bg: #fff3ee;
    --green:     #3ABC3F;
    --green-bg:  #F3FBF4;
    --green-text:#2E9632;
    --red:       #FF3131;
    --red-bg:    #FFF3F3;
    --red-text:  #CC2727;
    --yellow:    #FFA931;
    --yellow-bg: #FFFAF3;
    --yellow-text:#CC8727;
    --text:      #121212;
    --text-muted:#888888;
    --text-dim:  #b5b5b5;
    --input-bg:  #ffffff;
    --radius:    0px;
    --font:      'Inter', system-ui, sans-serif;
    --serif:     'Source Serif 4', Georgia, serif;
    --mono:      'IBM Plex Mono', monospace;
  }
```

- [ ] **Step 2: Verify**
```bash
grep -n "accent:\|radius:\|--mono:\|--serif" src/api/playground.html | head -10
```
Expected: `--accent: #ff4800`, `--radius: 0px`, `--mono: 'IBM Plex Mono'`, `--serif: 'Source Serif 4'`.

- [ ] **Step 3: Commit**
```bash
git add src/api/playground.html
git commit -m "style: replace :root tokens — orange accent, zero radius, new fonts"
```

---

## Task 3: Replace dark theme token block

**Files:**
- Modify: `src/api/playground.html:557-580`

- [ ] **Step 1: Replace the `[data-theme="dark"]` block**

Use Edit tool. Replace:
```css
  [data-theme="dark"] {
    --bg:        #0D0E0F;
    --surface:   #161718;
    --surface2:  #1E1F21;
    --border:    #2A2D30;
    --border2:   #252829;
    --accent:    #4D94FF;
    --accent-hi: #6BA8FF;
    --accent-lo: #1A5FCC;
    --accent-bg: #0D1F3C;
    --green:     #3ABC3F;
    --green-bg:  #0D2610;
    --green-text:#5BD460;
    --red:       #FF4D4D;
    --red-bg:    #2A0D0D;
    --red-text:  #FF6B6B;
    --yellow:    #FFA931;
    --yellow-bg: #2A1F0A;
    --yellow-text:#FFBE5C;
    --text:      #E8EAEC;
    --text-muted:#8B9299;
    --text-dim:  #5A6169;
    --input-bg:  #1E1F21;
  }
```
With:
```css
  [data-theme="dark"] {
    --bg:        #121212;
    --surface:   #1a1a1a;
    --surface2:  #222222;
    --border:    #2a2a2a;
    --border2:   #242424;
    --accent:    #ff4800;
    --accent-hi: #ff6a33;
    --accent-lo: #cc3a00;
    --accent-bg: #2a1200;
    --green:     #3abc3f;
    --green-bg:  #0d2410;
    --green-text:#4ecb54;
    --red:       #ff3131;
    --red-bg:    #2a0808;
    --red-text:  #ff5555;
    --yellow:    #ffa931;
    --yellow-bg: #2a1f00;
    --yellow-text:#ffb74d;
    --text:      #f0f0f0;
    --text-muted:#888888;
    --text-dim:  #555555;
    --input-bg:  #1a1a1a;
  }
```

- [ ] **Step 2: Replace the dark overrides below the block (lines ~581-584)**

Use Edit tool. Replace:
```css
  [data-theme="dark"] .response-box { background: #1a1b1d; }
  [data-theme="dark"] .chat-history { background: #1a1b1d; }
  [data-theme="dark"] .btn-ghost { background:#1E1F21; color:#E8EAEC; border-color:#2A2D30; }
  [data-theme="dark"] .btn-ghost:hover:not(:disabled) { background:#252829; }
```
With:
```css
  [data-theme="dark"] .response-box { background: #1a1a1a; }
  [data-theme="dark"] .chat-history { background: #1a1a1a; }
  [data-theme="dark"] .btn-ghost { background:#222222; color:#f0f0f0; border-color:#2a2a2a; }
  [data-theme="dark"] .btn-ghost:hover:not(:disabled) { background:#2a2a2a; }
```

- [ ] **Step 3: Verify**
```bash
grep -n "data-theme.*dark" src/api/playground.html | head -10
```
Expected: orange `#ff4800` accent in dark block, dark bg `#121212`, no `#4D94FF` or `#0D0E0F`.

- [ ] **Step 4: Commit**
```bash
git add src/api/playground.html
git commit -m "style: replace dark theme tokens — orange accent, deep-black palette"
```

---

## Task 4: Remove CSS box-shadows (flat design)

**Files:**
- Modify: `src/api/playground.html` (CSS block, lines ~128–615)

These are all `0px 3px 4px -4px` depth shadows and the inset button shadow. Remove them for flat design. Keep `box-shadow: none` (explicit resets), the `.ced-panel` dropdown shadow, and the `0 0 6px` / `0 0 4px` glow effects on status dots.

- [ ] **Step 1: Remove shadow from `.card`**

Replace:
```css
    border-radius: 12px; padding: 20px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
```
With:
```css
    border-radius: 12px; padding: 20px;
```

- [ ] **Step 2: Remove shadow + transition from `input[type=text], input[type=number], select`**

Replace:
```css
    outline: none; transition: border-color .15s, box-shadow .15s;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.15);
```
With:
```css
    outline: none; transition: border-color .15s;
```

- [ ] **Step 3: Update focus ring to orange on inputs**

Replace:
```css
  input[type=text]:focus, input[type=number]:focus, select:focus {
    border-color: var(--text-dim);
    outline: none;
    box-shadow: 0px 0px 0px 2px var(--border2);
  }
```
With:
```css
  input[type=text]:focus, input[type=number]:focus, select:focus {
    border-color: var(--accent);
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,72,0,.15);
  }
```

- [ ] **Step 4: Remove shadow + transition from `textarea`**

Replace:
```css
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.15);
    transition: border-color .15s, box-shadow .15s;
  }
  textarea:hover { border-color: var(--text-dim); box-shadow: none; }
```
With:
```css
    transition: border-color .15s;
  }
  textarea:hover { border-color: var(--text-dim); }
```

- [ ] **Step 5: Update textarea focus ring to orange**

Replace:
```css
  textarea:focus {
    border-color: var(--text-dim);
    outline: none;
    box-shadow: 0px 0px 0px 2px var(--border2);
  }
```
With:
```css
  textarea:focus {
    border-color: var(--accent);
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,72,0,.15);
  }
```

- [ ] **Step 6: Remove inset shadow from `.btn-primary`**

Replace:
```css
    background-color: #0D0E0F;
    color: #FFFFFF;
    box-shadow: inset 0px 2px 0px 0px rgba(255,255,255,0.25);
    border: 1px solid #0D0E0F;
```
With:
```css
    background-color: #0D0E0F;
    color: #FFFFFF;
    border: 1px solid #0D0E0F;
```

- [ ] **Step 7: Update `.btn-primary` hover gradient to orange**

Replace:
```css
    background: linear-gradient(90deg, #0D0E0F, #FF3131, #0066F5);
```
With:
```css
    background: linear-gradient(90deg, #0D0E0F, #FF3131, #ff4800);
```

- [ ] **Step 8: Remove shadow from `.btn-ghost`**

Replace:
```css
    background-color: #FFFFFF;
    border: 1px solid #E4E7E9;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.15);
    color: #0D0E0F;
```
With:
```css
    background-color: #FFFFFF;
    border: 1px solid #E4E7E9;
    color: #0D0E0F;
```

- [ ] **Step 9: Remove shadow from `.stat-card`**

Replace:
```css
    border-radius: 12px; padding: 14px 16px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
  }
  .stat-label {
```
With:
```css
    border-radius: 12px; padding: 14px 16px;
  }
  .stat-label {
```

- [ ] **Step 10: Remove shadow from `.tab-bar`**

Replace:
```css
    border-radius: 10px; padding: 3px; width: fit-content; margin-bottom: 20px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
```
With:
```css
    border-radius: 10px; padding: 3px; width: fit-content; margin-bottom: 20px;
```

- [ ] **Step 11: Remove shadow from `.dash-tile`**

Replace:
```css
    border-radius: 12px; padding: 14px 16px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
  }
  .dash-tile-label {
```
With:
```css
    border-radius: 12px; padding: 14px 16px;
  }
  .dash-tile-label {
```

- [ ] **Step 12: Remove shadow + transition from `.ced-trigger`**

Replace:
```css
    gap: 6px; transition: border-color .15s, box-shadow .15s; user-select: none;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.15);
  }
  .ced-trigger:hover { box-shadow: none; }
```
With:
```css
    gap: 6px; transition: border-color .15s; user-select: none;
  }
```

- [ ] **Step 13: Remove shadow from `.model-card`**

Replace:
```css
    border-radius: 12px; padding: 16px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
    display: flex; flex-direction: column; gap: 10px;
```
With:
```css
    border-radius: 12px; padding: 16px;
    display: flex; flex-direction: column; gap: 10px;
```

- [ ] **Step 14: Remove shadow from `.stat-tile`**

Replace:
```css
    border-radius: 12px; padding: 14px 16px;
    box-shadow: 0px 3px 4px -4px rgba(0,0,0,0.08);
    display: flex; flex-direction: column; gap: 4px;
```
With:
```css
    border-radius: 12px; padding: 14px 16px;
    display: flex; flex-direction: column; gap: 4px;
```

- [ ] **Step 15: Verify — no depth shadows remain in CSS**
```bash
grep -n "3px 4px -4px\|inset 0px 2px" src/api/playground.html
```
Expected: zero results.

- [ ] **Step 16: Commit**
```bash
git add src/api/playground.html
git commit -m "style: remove all depth box-shadows — flat design; update focus rings to orange"
```

---

## Task 5: Fix remaining CSS component styles

**Files:**
- Modify: `src/api/playground.html` (CSS block)

- [ ] **Step 1: Fix `.method.post` badge (blue → orange)**

Replace:
```css
  .method.post { background: #F0F6FE; color: #0052C4; }
```
With:
```css
  .method.post { background: #fff3ee; color: #cc3a00; }
```

- [ ] **Step 2: Fix `.dl-status-badge.downloading` (blue → orange)**

Replace:
```css
  .dl-status-badge.downloading { background: #F0F6FE; color: #0052C4; }
```
With:
```css
  .dl-status-badge.downloading { background: #fff3ee; color: #cc3a00; }
```

- [ ] **Step 3: Fix `.ced-item` hover and selected (blue rgba → orange rgba)**

Replace:
```css
  .ced-item:hover { background: var(--hover, rgba(88,166,255,.06)); }
  .ced-item.ced-selected { background: rgba(88,166,255,.1); color: var(--accent); }
```
With:
```css
  .ced-item:hover { background: var(--hover, rgba(255,72,0,.06)); }
  .ced-item.ced-selected { background: rgba(255,72,0,.1); color: var(--accent); }
```

- [ ] **Step 4: Fix `.badge` border (blue rgba → orange rgba)**

Replace:
```css
    border: 1px solid rgba(0,102,245,.2);
  }
  .dot {
```
With:
```css
    border: 1px solid rgba(255,72,0,.2);
  }
  .dot {
```

- [ ] **Step 5: Fix `.tab.active` border (blue rgba → orange rgba)**

Replace:
```css
    background: var(--accent-bg); color: var(--accent);
    border: 1px solid rgba(0,102,245,.2);
```
With:
```css
    background: var(--accent-bg); color: var(--accent);
    border: 1px solid rgba(255,72,0,.2);
```

- [ ] **Step 6: Fix `.ced-trigger.open` asymmetric radius**

Replace:
```css
  .ced-trigger.open { border-bottom-left-radius: 4px; border-bottom-right-radius: 4px; }
```
With:
```css
  .ced-trigger.open { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
```

- [ ] **Step 7: Fix `.ced-panel` bottom radius**

Replace:
```css
    border-top: none; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;
```
With:
```css
    border-top: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0;
```

- [ ] **Step 8: Fix `.system-prompt-field textarea` asymmetric radius**

Replace:
```css
  .system-prompt-field textarea { border-left: 3px solid var(--accent); border-radius: 0 12px 12px 0; }
```
With:
```css
  .system-prompt-field textarea { border-left: 3px solid var(--accent); border-radius: 0; }
```

- [ ] **Step 9: Add `font-family: var(--serif)` to `.panel-title`**

Replace:
```css
  .panel-title  { font-size: 18px; font-weight: 600; }
```
With:
```css
  .panel-title  { font-size: 18px; font-weight: 600; font-family: var(--serif); }
```

- [ ] **Step 10: Add `font-family: var(--serif)` to `.logo-name`**

Replace:
```css
  .logo-name { font-weight: 600; font-size: 15px; }
```
With:
```css
  .logo-name { font-weight: 600; font-size: 15px; font-family: var(--serif); }
```

- [ ] **Step 11: Verify**
```bash
grep -n "0052C4\|F0F6FE\|88,166,255\|0 12px 12px 0" src/api/playground.html
```
Expected: zero results.

- [ ] **Step 12: Commit**
```bash
git add src/api/playground.html
git commit -m "style: fix component-level blue colors, asymmetric radii, add serif to titles"
```

---

## Task 6: Update brand strings

**Files:**
- Modify: `src/api/playground.html:6-7, 808-809`

Run this task BEFORE the Python pass (Task 8) — the Python pass must not touch the favicon string before it is updated here.

- [ ] **Step 1: Update `<title>` tag**

Replace:
```html
<title>Kolosal Torch Inference</title>
```
With:
```html
<title>Netra RT — Torch Inference</title>
```

- [ ] **Step 2: Update favicon SVG (text KI→NR, rx=8→0, fill blue→orange)**

Replace:
```html
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'><rect width='32' height='32' rx='8' fill='%230066F5'/><text x='6' y='23' font-family='sans-serif' font-size='16' font-weight='700' fill='white'>KI</text></svg>" />
```
With:
```html
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'><rect width='32' height='32' rx='0' fill='%23ff4800'/><text x='4' y='23' font-family='sans-serif' font-size='16' font-weight='700' fill='white'>NR</text></svg>" />
```

- [ ] **Step 3: Update logo mark text in HTML body**

Replace:
```html
      <div class="logo-mark">KI</div>
```
With:
```html
      <div class="logo-mark">NR</div>
```

- [ ] **Step 4: Update logo name text**

Replace:
```html
      <span class="logo-name">Kolosal</span>
```
With:
```html
      <span class="logo-name">Netra RT</span>
```

- [ ] **Step 5: Update button comment in CSS (cosmetic)**

Replace:
```css
  /* Buttons — Kolosal vanilla */
```
With:
```css
  /* Buttons */
```

- [ ] **Step 6: Verify no "Kolosal" remains**
```bash
grep -in "kolosal" src/api/playground.html
```
Expected: zero results.

- [ ] **Step 7: Commit**
```bash
git add src/api/playground.html
git commit -m "brand: rename Kolosal → Netra RT; NR logo mark, orange favicon"
```

---

## Task 7: Fix JS hardcoded blue values

**Files:**
- Modify: `src/api/playground.html` (JS section, lines ~1509, ~2199-2211, ~4037, ~4249, ~4596)

- [ ] **Step 1: Fix CPU bar fill color (line ~1509)**

Replace:
```html
style="width:0%;background:#0066F5"
```
With:
```html
style="width:0%;background:#ff4800"
```

- [ ] **Step 2: Fix sparkline gradient (lines ~2199-2200)**

Replace:
```javascript
  grad.addColorStop(0, 'rgba(0,102,245,0.18)');
  grad.addColorStop(1, 'rgba(0,102,245,0.02)');
```
With:
```javascript
  grad.addColorStop(0, 'rgba(255,72,0,0.18)');
  grad.addColorStop(1, 'rgba(255,72,0,0.02)');
```

- [ ] **Step 3: Fix sparkline stroke/fill (lines ~2207, 2211)**

Replace:
```javascript
  ctx.strokeStyle = '#0066F5'; ctx.lineWidth = 1.5; ctx.stroke();
```
With:
```javascript
  ctx.strokeStyle = '#ff4800'; ctx.lineWidth = 1.5; ctx.stroke();
```

Find and replace the fill dot color on ~line 2211:
```bash
grep -n "fillStyle = '#0066F5'" src/api/playground.html
```
Then replace that line's `'#0066F5'` → `'#ff4800'` with Edit tool.

- [ ] **Step 4: Fix classification palette (line ~4037)**

Replace:
```javascript
  const palette = ['#FF3131','#0066F5','#3ABC3F','#FFA931','#CC27CC','#27CCCC','#CC7027','#2762CC'];
```
With:
```javascript
  const palette = ['#FF3131','#ff4800','#3ABC3F','#FFA931','#CC27CC','#27CCCC','#CC7027','#2762CC'];
```

- [ ] **Step 5: Fix detection palette (line ~4249)**

Replace:
```javascript
const DET_PALETTE = ['#FF3131','#0066F5','#3ABC3F','#FFA931','#CC27CC','#27CCCC'];
```
With:
```javascript
const DET_PALETTE = ['#FF3131','#ff4800','#3ABC3F','#FFA931','#CC27CC','#27CCCC'];
```

- [ ] **Step 6: Fix API Reference group color (line ~4596)**

Replace:
```javascript
    color: '#0066F5',
```
With:
```javascript
    color: '#ff4800',
```

- [ ] **Step 7: Verify no blue remains in JS**
```bash
grep -n "0066F5\|0052C4" src/api/playground.html
```
Expected: zero results.

- [ ] **Step 8: Commit**
```bash
git add src/api/playground.html
git commit -m "style: replace hardcoded blue in JS — sparkline, palettes, bar fill, API ref color"
```

---

## Task 8: Python pass — zero all remaining border-radius px values

**Files:**
- Modify: `src/api/playground.html` (entire file)

This pass zeros every `border-radius: Xpx` (single or multi-value) remaining in both CSS rules and inline HTML `style=""` attributes after the targeted edits above. Run AFTER Tasks 6 and 7.

- [ ] **Step 1: Run the replacement script**
```bash
python3 - <<'PYEOF'
import re

with open('src/api/playground.html', 'r') as f:
    s = f.read()

# Zero all border-radius values that use px (single or multi-value). Keep 50%.
def zero_radius(m):
    val = m.group(1)
    if '%' in val:
        return m.group(0)
    return 'border-radius:0'

s = re.sub(r'border-radius:([^;}\n"]+)', zero_radius, s)

# Mop up any remaining blue rgba values not caught by targeted edits
rgba_map = [
    ('rgba(0,102,245,0.18)', 'rgba(255,72,0,0.18)'),
    ('rgba(0,102,245,0.02)', 'rgba(255,72,0,0.02)'),
    ('rgba(0,102,245,.2)',   'rgba(255,72,0,.2)'),
    ('rgba(0,102,245,.25)',  'rgba(255,72,0,.25)'),
    ('rgba(88,166,255,.06)', 'rgba(255,72,0,.06)'),
    ('rgba(88,166,255,.1)',  'rgba(255,72,0,.1)'),
]
for old, new in rgba_map:
    s = s.replace(old, new)

with open('src/api/playground.html', 'w') as f:
    f.write(s)

print('done')
PYEOF
```

- [ ] **Step 2: Verify 50% radii are preserved**
```bash
grep -c "border-radius:50%" src/api/playground.html
```
Expected: > 0 (status indicator dots and spinner retain circular shape).

- [ ] **Step 3: Verify no blue rgba values remain**
```bash
grep -cn "0,102,245\|88,166,255" src/api/playground.html
```
Expected: `0`

- [ ] **Step 4: Commit**
```bash
git add src/api/playground.html
git commit -m "style: zero all remaining px border-radii; mop up blue rgba values"
```

---

## Task 9: Build and verify

**Files:**
- Read: `src/api/playground.html`

- [ ] **Step 1: Final colour audit**
```bash
grep -cn "0066F5\|0052C4\|F0F6FE\|4D94FF\|1A5FCC\|0D1F3C\|Kolosal\|Geist" src/api/playground.html
```
Expected: `0`

- [ ] **Step 2: Confirm orange and new tokens are present**
```bash
grep -c "ff4800" src/api/playground.html
```
Expected: ≥ 15 occurrences.

- [ ] **Step 3: Build the server**
```bash
cargo build --release 2>&1 | tail -5
```
Expected: `Finished release [optimized] target(s)` with no errors.

- [ ] **Step 4: Start server and open playground in browser**
```bash
./target/release/torch-inference-server &
open http://localhost:8000/playground
```
Visually verify:
- Logo mark shows `NR` in an orange square
- Topbar shows "Netra RT" in serif font
- All cards and inputs have square (0px) corners
- No blue anywhere; accent color is orange
- Dark mode toggle still works and shows the dark orange palette

- [ ] **Step 5: Kill server**
```bash
pkill torch-inference-server
```

- [ ] **Step 6: Final commit if any cleanup needed, otherwise done**
```bash
git log --oneline -8
```
