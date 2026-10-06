PROJECT NAME: distributed_training_pytorch

META-INSTRUCTIONS:

<Read it all before acting. Ask about anything unclear, contradictory or
 underspecified — before starting and mid-build. Ask in the question widget
 (AskUserQuestion): related questions batched, concrete options, your
 recommendation first. Plain text only if the widget isn't available.>

<Don't expand scope. Anything not listed here is a proposal, including changes
 to this file — propose it, don't do it.>

<Prefer doing over describing: run the code, write the files, test it.>

<Always in scope, no proposal needed: when it goes on GitHub, a README that is
 easy to read at a glance — a line on what it is, then clear visuals
 (screenshots, a diagram or a chart), then links. Everything else goes in
 linked files: docs/INSTRUCTIONS.md (setup, run, use),
 docs/SYSTEM-DESIGN.md (see below) and docs/FILE-STRUCTURE.md (what's where). Also a small unobtrusive feedback tab
 if what you're building is an application rather than a script.>

<If what you're building is an application, build it as a Mac app first; the
 website comes after, as its own step.>

<Name things the way a person would say them — "Goal Tracker", not
 goal_tracker — for the app, its windows, titles, files people open, repo
 descriptions and README headings. When you create the GitHub repo, name it
 with no "_" or "-": one word or joined words, e.g. GoalTracker.>

<Always in scope: a system design doc in the codebase, docs/SYSTEM-DESIGN.md,
 kept current as the build changes. Cover the architecture (with a Mermaid
 diagram), each component's job, the main flows, where data lives, the key
 design decisions and their trade-offs, how it's tested, and known limits.>

<Finish by listing every deliverable: path, what it is, how to check it works.>

<Git rules (no Claude attribution, never commit .claude/) are in
 ~/.claude/CLAUDE.md and apply on their own — nothing to repeat here.>

<Keep the changelog at the bottom current.>

CONTEXT:

create a demo for generalized distributed training in pytorch

DELIVERABLES:

demo. site

OPEN QUESTIONS / ASSUMPTIONS:

<Agent fills in: what it guessed, what it decided without asking.>

Asked and answered (2026-10-04):
- Scope: single + DDP + FSDP + pipeline parallel behind one --strategy flag (no tensor parallel).
- Site: static explainer with animated diagrams + charts of real runs, on GitHub Pages.
- GitHub: public repo "DistributedTraining" with Pages.
- Environment: project-local uv venv, Python 3.11.

Decided without asking:
- It's a script/library plus a static site, not an application, so no Mac app and no feedback tab.
- PyTorch 2.2.2: the newest build for Intel Macs. So FSDP uses the FSDP1 API and the pipeline
  is a hand-written GPipe schedule (torch.distributed.pipelining and FSDP2 need 2.4+).
- Runs on CPU with Gloo; GPU/NCCL and multi-node paths are written but untested.
- Data is a synthetic Markov chain (no downloads, known best-possible loss).
- Every strategy is built to match single-process training exactly; tests and the benchmark check it.
- Communication volume on the site is a formula, not a measurement (Gloo has no byte counters).
- No combined strategies — proposal, not built.
- Mixed precision (asked 2026-10-05): --precision fp32|bf16|fp16 via autocast, weights stay fp32;
  site compares bf16 with fp32. fp16 is GPU-only and untested here; pipeline rejects fp16.
- Checkpoints (asked 2026-10-04): one portable file any strategy can resume from; site shows a
  run saved under DDP x4 and finished under FSDP x2.

CHANGELOG:

- 2026-10-04 — created
- 2026-10-05 — added mixed precision training (--precision), tests, site chart
- 2026-10-04 — added checkpoint save and resume (portable across strategies), tests, site chart
- 2026-10-04 — built: disttrain package (single/DDP/FSDP/pipeline), tests, benchmark, site on GitHub Pages; filled in assumptions
- 2026-09-15 — added meta-instruction: built-out applications include a small feedback tab
- 2026-09-15 — added meta-instruction: no "Claude" attribution in commits, PRs, or branches
- 2026-09-16 — added meta-instruction: always include a README when adding to GitHub
- 2026-09-16 — changed meta-instruction: ask clarifying questions in the question widget
- 2026-09-17 — added meta-instructions: Claude never a contributor; never commit .claude/
- 2026-09-26 — compressed the meta-instructions and every field prompt; git rules moved to the global instruction file
- 2026-09-27 — added meta-instruction: applications are built as a Mac app first, then a website
- 2026-09-28 — folded inputs, instructions, constraints, deliverables and done criteria into one free-form CONTEXT
- 2026-09-28 — changed meta-instruction: a README on GitHub always includes a visual
- 2026-09-28 — added meta-instruction: name things like a person would, never snake_case
- 2026-09-28 — changed meta-instruction: README leads with visuals; instructions live in a linked guide
- 2026-09-28 — changed meta-instruction: README is visuals and links; details in docs/INSTRUCTIONS.md and docs/FILE-STRUCTURE.md
- 2026-09-29 — changed meta-instruction: GitHub repo names have no "_" or "-"
- 2026-10-02 — added a DELIVERABLES field after CONTEXT
- 2026-10-02 — added meta-instruction: every project has a system design doc at docs/SYSTEM-DESIGN.md
