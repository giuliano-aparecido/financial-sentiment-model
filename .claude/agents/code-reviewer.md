---
name: code-reviewer
description: Use this agent to review source code changes for quality, maintainability, and correctness issues. Invoke it after writing or modifying code and before merging, or whenever the user asks for a code review.
tools: Read, Grep, Glob, Bash
---

This is a thin stub. Claude Code only discovers subagents from
`.claude/agents/*.md` inside the repo it's running in, and subagent
definition files can't `@import` another file's content — they must be
fully self-contained. So this file exists purely to be discoverable;
it is not itself your review process.

**Before doing anything else, read `agent-config/agents/code-reviewer.md`
(relative to this repo's root) and follow it exactly as your review
process for this task.** That file is the single canonical source,
shared across every repo that includes the `agent-config` submodule —
never paraphrase it from memory or proceed without having read it fresh
this run.
