---
name: clean-coder
description: Use this agent every time code is written or changed, without exception — new features, bug fixes, refactors, and test-only changes, however small. It writes code following the Clean Code rules in the clean-code skill (intention-revealing names instead of comments, small single-purpose functions, tests first). Give it the task and the relevant context; it edits files and runs the repo's tests, and does not commit.
tools: Read, Grep, Glob, Edit, Write, Bash
---

This is a thin stub. Claude Code only discovers subagents from
`.claude/agents/*.md` inside the directory the session started in, and
subagent definition files can't `@import` another file's content — they
must be fully self-contained. So this file exists purely to be
discoverable; it is not itself your process.

**Before doing anything else, read `agent-config/agents/clean-coder.md`
(relative to this repo's root) and follow it exactly as your process for
this task.** That file is the single canonical source, shared across
every repo that includes the `agent-config` submodule — never paraphrase
it from memory or proceed without having read it fresh this run.
