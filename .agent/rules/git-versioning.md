# Strict Automated Git Versioning & Remote Push Protocol

## Objective
Prevent any accidental data loss, truncation, or corruption across all sessions by enforcing strict, automated version control tracking and remote synchronization with GitHub.

## Rules & Protocol

### 1. Pre-Change Safeguards
- Prior to making any edit to primary paper sources (`manuscript.tex`, `sample-base.bib`, figures, tables, scripts, or core model files), verify that the working tree is clean via `git status`.
- **No Greedy Dot-All Replacements**: Never execute regex replacement scripts using `re.compile(..., re.DOTALL)` across generic multiline tags like `\begin{figure*}`. Always use exact string replacement or tightly anchored blocks.
- **No PowerShell Math Interpolation**: Never pass LaTeX text containing math delimiters (`$`) through PowerShell inline `-c "..."` commands, which interpolate `$1`, `$G`, `$OA` into blank strings. Always write temporary scripts using file tools or execute clean python files.

### 2. Immediate Post-Change Git Commit & Push
- Immediately following every successful file update, figure generation, table edit, or PDF compilation pass:
  1. Check modified files: `git status`.
  2. Stage changes: `git add .` (respecting `.gitignore`).
  3. Commit with a meaningful message: `git commit -m "<type>(scope): <description>"`.
  4. Push directly to remote: `git push origin main`.

### 3. Remote Repository Target
- **Remote**: `origin` (`https://github.com/codeforlifeee/paper.git`)
- **Branch**: `main`
- Ensure that every commit is successfully synced to the remote so the user can inspect diffs, track full history, or rollback anytime.

