# AGENTS.md - Workspace Agent Instructions

## Critical Operational Rules

### 1. Automated Git Versioning & Immediate Push Protocol
- **Zero Data Loss Guarantee**: All changes made to `manuscript.tex`, figures, tables, code, or documentation must be committed and pushed to GitHub (`origin/main`) immediately after each successful task/turn.
- **Workflow**:
  1. `git status` to verify modified files.
  2. `git add .` (obeying `.gitignore`).
  3. `git commit -m "feat/fix/docs/refactor: <clear description>"`.
  4. `git push origin main`.
- **Pre-edit Safeguard**: Before modifying `manuscript.tex`, always ensure working tree is committed so that any change can be diffed with `git diff` or reverted with `git checkout`.
- **Safe Text Replacement**: Never use greedy `re.DOTALL` regexes over broad blocks like `\begin{figure*}`. Always use exact unique string replacements.
- **No PowerShell Math Corruption**: Do not invoke python inline commands through PowerShell containing `$` math symbols, as PowerShell eats `$1` or `$G`. Write a script file and run it.

### 2. Paper Benchmark Master Truth
- Always strictly cite and cross-verify all quantitative numbers against:
  - `README_INDIAN_PINES.md` (Indian Pines baselines)
  - `README.md` (Pavia University & Salinas Scene baselines)
  - `MASTER_README_120+runs_SGSCF_Net.html` (SGSCF-Net 60 grid search runs, 12 scarcity runs, and ablations)
