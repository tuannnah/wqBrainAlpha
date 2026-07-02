---
name: session-journal
description: >-
  Maintain and read a running development progress log (PROGRESS.md) for the MiniBrain
  project so work carries across sessions, since each session starts with no memory of the
  last. Use this skill at the START of any MiniBrain working session to read what previous
  sessions accomplished, the current phase, open blockers, and the planned next step before
  doing anything else; and at the END of a session — or after finishing a phase or any
  meaningful chunk of work — to append a dated, structured entry recording what was done,
  decisions made, what is in progress, blockers, and the next step. Trigger whenever
  resuming, continuing, wrapping up, checking status, or reporting progress on MiniBrain —
  or when the user says things like "where were we", "what's left", "continue", "let's
  start", or "log this" — even if they do not explicitly mention a journal or progress log,
  so progress is always tracked and no context is lost between sessions.
---

# Session journal

Sessions are stateless: a new session has no memory of what the previous one did. The
journal (`PROGRESS.md` at the repo root) is the durable, append-only memory of progress for
the MiniBrain build. **Read it first, write to it last.**

## Where it lives
`PROGRESS.md` at the repository root. If it does not exist yet, create it from the template
in `references/entry-template.md` (a `Current state` block at the top, then an `Entries`
section). Keep the whole log in this one file.

## Session-start ritual (do this before any work)
1. Read the **`Current state`** block at the top of `PROGRESS.md` — current phase, what is
   done, what is in progress, the next planned step, and any open blockers.
2. Read the **last 1–3 entries** for recent decisions and context.
3. Briefly tell the user where things stand and what the next step is, then proceed. If the
   user's request conflicts with the recorded next step, surface that before diverging.

## Session-end / per-phase ritual (do this when wrapping up or finishing a phase)
1. **Append** a new dated entry to the `Entries` section using the template format below.
   Entries are append-only — never edit or delete past entries (they are the history).
2. **Update** the `Current state` block at the top so it reflects reality after this
   session (phase, done, in-progress, next step, blockers). This block is the one part of
   the file that is rewritten each time; everything under `Entries` only grows.
3. Keep entries **concise but specific**: name the files touched, the decisions made (and
   why), what is verified vs. still open, and the single most important next step.

## Entry format
Use this exact structure (see `references/entry-template.md` for a filled-in example):

```
### [YYYY-MM-DD] Session NN — <short title>
- **Phase:** <phase number/name and where within it>
- **Done:** <what was completed and verified this session>
- **Decisions:** <choices made and the reason; deviations from the design + why>
- **In progress:** <started but not finished>
- **Blockers / open risks:** <anything blocking, or risks to watch — esp. look-ahead bias,
  operator fidelity, low calibration ρ>
- **Next step:** <the single most important thing to do next>
- **Tests:** <suite status — green/red, and what was added>
```

## Rules
- **Append-only entries; live `Current state`.** Never rewrite history; always refresh the
  top block.
- **One entry per meaningful chunk** — a finished phase, a significant decision, or the end
  of a working session. Don't spam tiny entries; don't batch a whole phase into a vague one.
- **Tie every entry to the phase plan** (`minibrain-builder` → `references/phases.md`) and
  its Definition of Done, so "done" is unambiguous.
- **Record decisions, not just status.** The reason behind a choice is the thing a future
  session can't reconstruct.
- **Be honest about state.** Mark things as verified only when tests prove them; otherwise
  they go under `In progress` or `Blockers`.

This skill pairs with `minibrain-builder`, whose per-phase ritual ends with "update the
journal." Reading and updating the journal is not optional overhead — it is how the build
stays coherent across many short sessions.
