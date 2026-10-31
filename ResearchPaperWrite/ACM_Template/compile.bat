@echo off
setlocal enabledelayedexpansion
set "PATH=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64;%PATH%"

echo =======================================================
echo Compiling ACM Manuscript with MiKTeX...
echo =======================================================

echo [Pass 1/3] Generating initial auxiliary files...
pdflatex --enable-installer -interaction=nonstopmode manuscript.tex > nul

echo [Pass 2/3] Processing BibTeX bibliography citations...
bibtex manuscript > nul

echo [Pass 3/3] Resolving cross-references and building final PDF...
pdflatex --enable-installer -interaction=nonstopmode manuscript.tex > nul
pdflatex --enable-installer -interaction=nonstopmode manuscript.tex

if exist manuscript.pdf (
    echo.
    echo =======================================================
    echo SUCCESS: manuscript.pdf has been built successfully!
    echo =======================================================
) else (
    echo.
    echo ERROR: Compilation failed. Check manuscript.log for details.
    exit /b 1
)
exit /b 0
