# GEMINI.md - Workspace Instructions for Antigravity & Gemini Agents

## Strict Git Versioning & Automated Cloud Sync Protocol

- **Immediate Commit & Push**: Whenever files in this project (especially `manuscript.tex`, figures, tables, scripts, or core model files) are created or updated, immediately stage, commit, and push to GitHub:
  ```bash
  git add .
  git commit -m "<type>(scope): <description>"
  git push origin main
  ```
- **Safe Modifications**: Never use greedy multiline regex replacements across multiple figures/sections. Always use exact string replacement.
- **PowerShell Protection**: Never run inline python strings containing `$` math formulas inside PowerShell double-quotes, as PowerShell expands `$1`, `$G`, etc. to empty strings. Always write temporary scripts via dedicated tools.
