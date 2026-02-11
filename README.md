# README — Dueffe Quilting CNC (Two-Head: Y & Z) File Format

This is chatgpt generated, based on my descriptions.

## 0) Machine model & axes

This CNC dialect targets a **Dueffe quilting machine** with:

* **Shared X axis**
* **Two sewing heads** positioned on the other axis:

  * **Head Y** (referred to as “Y”)
  * **Head Z** (referred to as “Z”)

A program can run:

* **Single-head mode** (Y *or* Z)
* **Dual-head mode** (both heads active), in either:

  * **Parallel mode** (`DW13`)
  * **Mirror mode** (`DW14`)

---

## 1) File layout (high level)

A full CNC file has these major parts, in this order:

1. **Speed blocks**: `BLOCK VA1/VA2/VA3 ... ENDBL`
2. **Program header**: `PROGRAM ...` + ABS/VEL/ACC + init calls
3. **One or more “sections”**, each typically:

   * (optional) lock axis (`FLZ` / `FLY`)
   * distance set (`QLY` / `QLZ` / `QLYZ`)
   * positioning move without sewing (`MR`)
   * motor enable (`ELY` / `ELZ` / `ELYZ`) *only when state changes*
   * start sewing (`DW11/12/13/14`)
   * geometry (`MI`, `MOVI`, `ARC`, with `FREEZE`/`SYNC`)
   * end sewing (`UP1`)
4. **Finish block** (always the same): `STOFF ... ENDPR`

---

## 2) Speed blocks (VA1 / VA2 / VA3)

Example (from your file):

```cnc
BLOCK VA1
VELL 200
ACCL 600
w195=1
ENDBL
```

* These define preset velocity/acceleration profiles.
* They are called later via `CALL VA1` / `CALL VA2` / `CALL VA3`.
* In practice, they’re **static boilerplate** and rarely change.

---

## 3) Program header (standard boilerplate)

Example:

```cnc
PROGRAM TEST_DRUGE_GLAVE
; DISEGNO: TEST_DRUGE_GLAVE
ABS X=0
ABS Y=0
CORNER a=333.33
VEL X= 80
VEL Y= 80
ACC X= 30
ACC Y= 30
v990=1
v991=1
CALL INIZIO
LABEL 1
CALL INLAV1
CALL VA1
```

Conventions:

* The `; DISEGNO:` line is a human-readable design label.
* `ABS`, `VEL`, `ACC` set job-level motion behavior.
* Initialization macros (`INIZIO`, `INLAV1`) are always present.
* Header typically ends with `CALL VA1`.

---

## 4) Head selection / axis locking: FLZ vs FLY vs none

### 4.1 Lock commands (single-head mode)

* `CALL FLZ` = **freeze-lock Z** → **Y head runs alone**
* `CALL FLY` = **freeze-lock Y** → **Z head runs alone**
* If neither is called → **both heads active**

### 4.2 Distance set macros (must follow lock choice)

Immediately after choosing head mode, set the “distance”:

* Y-only (after `FLZ`): `CALL QLY <yDist>`
* Z-only (after `FLY`): `CALL QLZ <zDist>`
* Dual-head: `CALL QLYZ <yDist> <zDist>`

**Rule:** `QLY/QLZ/QLYZ` is **always followed by** a **non-sewing positioning move** (`MR ...`).

Examples:

**Y-only:**

```cnc
CALL FLZ
CALL QLY 963
MR X53.74Y963
```

**Dual-head:**

```cnc
CALL QLYZ 613 1213
MR X587.34Y613Z1237
```

> Naming note: “QLY/QLZ/QLYZ” meaning is unknown, but in every file they behave like a required *offset/distance configuration* for the active head(s).

---

## 5) Motor enable macros (ELY / ELZ / ELYZ)

These are called **when the motor state changes**:

* `CALL ELY` → enable Y motor
* `CALL ELZ` → enable Z motor
* `CALL ELYZ` → enable both motors

Typical pattern: after the (lock → QL* → MR) setup block:

```cnc
CALL FLZ
CALL QLY 963
MR X53.74Y963
CALL ELY
```

**Important:** Many files repeat `CALL ELYZ` again later before another dual-head block. Treat that as “re-assert motor state” (safe), even if it appears redundant.

---

## 6) Start sewing macros (DW11 / DW12 / DW13 / DW14)

These lower head(s) and begin stitching. They must match the current motor/head mode:

* `CALL DW11` → **Y head only**

  * Expected prior: `FLZ` + `ELY`
* `CALL DW12` → **Z head only**

  * Expected prior: `FLY` + `ELZ`
* `CALL DW13` → **dual-head, parallel**

  * Expected prior: `ELYZ`
* `CALL DW14` → **dual-head, mirror**

  * Expected prior: `ELYZ`

Examples from your file:

* Single Y head circles: `CALL DW11 ... ARC ... SYNC ...`
* Dual-head mirrored circles: `CALL DW14 ... ARC ...`
* Dual-head parallel polylines: `CALL DW13 ... MI ...`

---

## 7) End sewing macro: UP1

`CALL UP1` raises head(s) and ends the stitching segment (“pen up”).

**Rule:** Every `DWxx` block should terminate with `CALL UP1` before changing modes, distances, or doing unrelated positioning.

---

## 8) Geometry commands & motion quirks

All geometry moves start from the **current position** and end at the coordinate provided.

### 8.1 MI vs MOVI (straight lines)

Two straight line commands exist:

* `MI X...Y...` / `MI X...Z...`
* `MOVI X...Y...` / `MOVI X...Z...`

Known quirk (important):

* **You cannot call `FREEZE` immediately after `MOVI`.**
* `FREEZE` is allowed after `MI`.

Example from your file:

```cnc
MI X655.73Y0
FREEZE
MOVI X655.73Y975
MI X655.73Y1950
```

### 8.2 ARC (arcs)

Format:

```cnc
ARC X<endX>Y<endY> a=<angle>
```

* Angle sign controls direction:

  * `a < 0` clockwise
  * `a > 0` counterclockwise

### 8.3 Mixing rules (hard constraints you observed)

* `ARC` + `MI` can be mixed, **but do not put `MI` between consecutive `ARC` commands** (keep ARC chains contiguous).
* `ARC` + `MOVI` can be mixed, and `MOVI` **may** appear between arcs.

### 8.4 FREEZE

`FREEZE` appears frequently as a boundary between moves/segments.

Practical usage (observed):

* Used before arcs and between groups of moves, likely to force the controller to “commit” the current segment(s) and avoid blending or axis issues.
* Avoid placing it right after `MOVI`.

### 8.5 SYNC (ARC-related)

**Observation:** `SYNC` is “ARC-relevant” and usually appears after an arc chain, before finishing the stitch block or resuming with MI.

Examples:

```cnc
ARC ...
ARC ...
ARC ...
SYNC
MI ...
CALL UP1
```

Working hypothesis (documented behavior, not proven):

* `SYNC` likely forces motion planner / heads to finish queued arc interpolation and align state before continuing (especially important with dual-head arcs).

---

## 9) Section patterns from the full file (annotated)

### 9.1 Y-only “triple-arc circle-ish” pattern

```cnc
CALL ELY
CALL DW11
MI X53.74Y975
FREEZE
ARC X153.74Y975 a=180
ARC X53.74Y975 a=180
ARC X153.74Y975 a=180
SYNC
MI X153.74Y963
CALL UP1
```

Notes:

* Starts with a short `MI`, then `FREEZE`, then a chain of arcs.
* Ends with `SYNC` and a final `MI` before `UP1`.

### 9.2 Dual-head mirror circles (DW14)

```cnc
CALL ELYZ
CALL DW14
MI X587.34Y625
FREEZE
ARC X487.34Y625 a=-180
ARC X587.34Y625 a=-180
ARC X487.34Y625 a=-180
SYNC
MI X487.34Y613
CALL UP1
```

### 9.3 Dual-head polyline (DW13)

```cnc
CALL DW13
MI X78.69Y175.35
MI X150.8Y175.35
MI X150.8Y130.43
MI X69.23Y130.43
MI X69.23Y183.63
MI X69.23Y171.63
CALL UP1
```

No arcs → no SYNC needed.

### 9.4 Frame/perimeter with MOVI (FREEZE placement)

```cnc
CALL DW11
MI X0Y29.43
MI X0Y0
MI X655.73Y0
FREEZE
MOVI X655.73Y975
MI X655.73Y1950
MI X0Y1950
FREEZE
MOVI X0Y975
MI X0Y0
MI X0Y12
CALL UP1
```

---

## 10) Finish block (always identical)

This block appears at the end and does not change:

```cnc
CALL STOFF
MR X=v993 Y=v994
CALL FINLAV1
CALL FINECIC1
IF (w92=1) JUMP 1
CALL FINE
ENDPR
```

Documented behavior:

* Stops sewing (`STOFF`)
* Returns to stored coordinates (`v993`, `v994`)
* Runs standard finish macros and ends the program.

---

## 11) “Hard rules” checklist (use for validation)

Before shipping a CNC file, check:

* [ ] File begins with `BLOCK VA1/VA2/VA3` definitions.
* [ ] Header contains init calls and ends with `CALL VA1` (or intentional VA preset).
* [ ] If `FLZ` appears → next config is `QLY`, and the next move is `MR ...Y...`
* [ ] If `FLY` appears → next config is `QLZ`, and the next move is `MR ...Z...`
* [ ] If no lock command → use `QLYZ ... ...` and `MR ...Y...Z...`
* [ ] Motor enable (`ELY/ELZ/ELYZ`) matches the upcoming `DWxx` mode.
* [ ] Every `DWxx` has a matching `CALL UP1`.
* [ ] `FREEZE` never appears immediately after `MOVI`.
* [ ] ARC chains usually end with `SYNC` before `UP1` or before switching back to line moves.
* [ ] File ends with the exact finish block.

---
