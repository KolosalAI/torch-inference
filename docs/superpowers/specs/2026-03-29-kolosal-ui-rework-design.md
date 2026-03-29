# Kolosal Vanilla UI Rework — Design Spec
Date: 2026-03-29

## Overview

Rework `src/api/playground.html` (1532-line single-file embedded dashboard) from a custom dark indigo theme to the **Kolosal vanilla design system** (light theme). Source: [KolosalAI/kolosal-library-vanilla](https://github.com/KolosalAI/kolosal-library-vanilla).

Approach: **Option A — Full 1:1 design token translation**. Translate all CSS vars and component classes to Kolosal vanilla tokens in plain CSS. No build step, no SCSS compilation. File stays a single self-contained HTML embedded in Rust via `include_str!`.

---

## 1. Color & Typography Tokens

All dark theme CSS vars replaced with Kolosal vanilla light palette:

| Role | Old | New (Kolosal token) |
|---|---|---|
| Page background | `#08080f` | `#F8F9F9` — `color-grey-500` |
| Surface (sidebar/topbar) | `#0e0e1a` | `#FFFFFF` — `color-neutral-white` |
| Surface 2 (cards/inputs) | `#13131f` | `#FFFFFF` with `#E4E7E9` border |
| Border primary | `#1f1f32` | `#DDE1E3` — `color-grey-900` |
| Border secondary | `#2a2a42` | `#E4E7E9` — `color-grey-800` |
| Text primary | `#e2e8f0` | `#0D0E0F` — `color-text-900` |
| Text muted | `#8892a4` | `#6A6F73` — `color-text-700` |
| Text dim | `#4b5568` | `#9C9FA1` — `color-text-600` |
| Text placeholder | — | `#9C9FA1` — `color-text-600` |
| Accent/primary | `#6366f1` indigo | `#0066F5` — `color-information-500` |
| Accent bg | `rgba(99,102,241,.08)` | `#F0F6FE` — `color-information-100` |
| Accent hi | `#818cf8` | `#3C8AF7` — `color-information-400` |
| Accent lo | `#4f46e5` | `#003D93` — `color-information-700` |
| Success/green | `#22c55e` | `#3ABC3F` — `color-success-500` |
| Danger/red | `#ef4444` | `#FF3131` — `color-danger-500` |
| Warning/yellow | `#f59e0b` | `#FFA931` — `color-warning-500` |

**Typography:**
- Body font: Inter (Google Fonts, already loaded) — `font-size: 14px`, `line-height: 1.6`, `letter-spacing: -0.2px`
- Mono font: Geist Mono (replace JetBrains Mono) — loaded via Google Fonts
- Font smoothing: `-webkit-font-smoothing: antialiased`
- Icons: **Remix Icons** v4.7.0 via `https://cdn.jsdelivr.net/npm/remixicon@4.7.0/fonts/remixicon.css`

---

## 2. Layout & Navigation

**Shell grid**: Unchanged — `220px sidebar + 1fr main`, `52px topbar + 1fr`.

**Topbar:**
- Background: `#FFFFFF`
- Border bottom: `1px solid #DDE1E3`
- Logo mark: black square (`#0D0E0F` bg), white "KI" text, `border-radius: 8px`
- Health badge: `.badge-sm` styled with information/success/danger color families

**Sidebar:**
- Background: `#FFFFFF`
- Border right: `1px solid #DDE1E3`
- Section labels: `#9C9FA1`, 10px uppercase, `letter-spacing: 0.08em`
- Nav item default: `color: #6A6F73`, transparent bg, `border-radius: 8px`
- Nav item hover: bg `#F8F9F9` (grey-500)
- Nav item active: bg `#F0F6FE` (information-100), `color: #0066F5` (information-500)
- Nav icon: `font-size: 16px`, `width: 18px`

**Remix Icon mapping for nav:**
| Panel | Icon class |
|---|---|
| Status | `ri-pulse-line` |
| TTS Stream | `ri-music-2-line` |
| Classify | `ri-image-line` |
| LLM Chat | `ri-chat-3-line` |
| Completion | `ri-edit-line` |
| Dashboard | `ri-dashboard-3-line` |
| Endpoints | `ri-link-m` |

**Main content area:**
- Background: `#F8F9F9`
- Padding: `28px 32px`
- Panel gap: `20px`
- Panel title: `font-size: 18px`, `font-weight: 600`, `color: #0D0E0F`
- Panel desc: `font-size: 13px`, `color: #6A6F73`

---

## 3. Components

### Cards
- Background: `#FFFFFF`
- Border: `1px solid #E4E7E9`
- Border radius: `12px`
- Padding: `20px`
- Shadow: `0px 3px 4px -4px rgba(0,0,0,0.10)`
- Card title: `11px`, uppercase, `color: #9C9FA1`, `letter-spacing: 0.06em`, `margin-bottom: 14px`

### Buttons
Kolosal `btn-md` (36px height, `border-radius: 10px`) with variants:

| Usage | Class |
|---|---|
| Primary actions (Synthesise, Send, Complete, Classify) | `btn-md btn-primary` (black bg, white text, gradient hover) |
| Secondary/cancel (Stop, Clear, Refresh) | `btn-md btn-outline` (white bg, grey border) |
| Destructive | `btn-md btn-danger` |

Button styles verbatim from Kolosal — including the `inset 0px 2px 0px 0px rgba(255,255,255,0.25)` inner highlight and the red→blue gradient hover on primary.

### Inputs
All `input[type=text]`, `input[type=number]`, `select` → `input-text-md` style:
- Height: `36px`, padding: `0 14px`
- Border: `1px solid #DDE1E3`, radius: `10px`
- Shadow: `0px 3px 4px -4px rgba(0,0,0,0.15)`
- Focus: border `#9C9FA1`, outline `0px 0px 0px 2px #E4E7E9`
- Placeholder: `#9C9FA1`

Textareas → `input-textarea` style:
- Padding: `8px 12px`, min-height: `156px` (or custom per panel)
- Border radius: `12px`

### Response / Output Boxes
- Background: `#F1F3F4` (grey-600)
- Border: `1px solid #E4E7E9`
- Border radius: `10px`
- Padding: `14px`
- Font: Geist Mono, `12.5px`, `line-height: 1.7`
- Default text: `#6A6F73`
- `.ok` state: `#0D0E0F`
- `.error` state: `#FF3131`
- `.stream` state: `#3ABC3F`

### Badges (status indicators)
`.badge-sm` pattern — `height: 24px`, `border-radius: 6px`, `padding: 0 8px`:
- Information: bg `#F0F6FE`, text `#0052C4`
- Success: bg `#F3FBF4`, text `#2E9632`
- Danger: bg `#FFF3F3`, text `#CC2727`
- Warning: bg `#FFFAF3`, text `#CC8727`

### Dashboard Tiles
- White card bg, border `1px solid #E4E7E9`, radius `10px`, padding `14px 16px`
- Tile label: `10px`, uppercase, `#9C9FA1`
- Tile value: `22px`, `font-weight: 700`, Geist Mono, `color: #0066F5` default
- Tile sub: `10px`, `#9C9FA1`

### Progress Bars (resource + download)
- Track: bg `#EBEDEE` (grey-700), `height: 6px`, radius `3px`
- Fill default: `#0066F5` (information-500)
- Fill done: `#3ABC3F` (success-500)
- Fill failed: `#FF3131` (danger-500)
- Fill warning: `#FFA931` (warning-500)

### Chat Messages
- User bubble: bg `#0066F5`, white text
- Assistant bubble: bg `#FFFFFF`, border `1px solid #E4E7E9`, text `#0D0E0F`
- Role label: `10px`, uppercase, `#9C9FA1`

### Method Badges (Endpoints panel)
- GET: bg `#F3FBF4`, text `#2E9632`
- POST: bg `#F0F6FE`, text `#0052C4`

### Dropzone (Image upload)
- Border: `2px dashed #DDE1E3`, radius `10px`, padding `28px`
- Text: `#9C9FA1`
- Hover: border `#0066F5`, bg `#F0F6FE`

---

## 4. Files Changed

| File | Change |
|---|---|
| `src/api/playground.html` | Full CSS rework — tokens, components, icons, fonts |

No new files. No Rust changes. No build pipeline changes.

---

## 5. Out of Scope

- Dark mode
- Responsive/mobile layout
- Any JS behavior changes
- Adding new panels or features
