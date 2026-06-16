from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
ZIP_NAME = "manuscript_no_figures_single_tex_20260616.zip"


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def flattened_main_tex() -> str:
    expanded = subprocess.run(
        [
            "latexpand",
            "--keep-comments",
            "biophysj.tex",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    expanded = re.sub(
        r"^[ \t]*\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}[ \t]*\n",
        "",
        expanded,
        flags=re.MULTILINE,
    )
    expanded = expanded.replace("\\section{Supplementary Material}", "\\section{Supplemental Material}")
    return expanded


def prepare_staging(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    run(
        [
            "xelatex",
            "-interaction=nonstopmode",
            f"-output-directory={staging}",
            "biophysj_SI.tex",
        ],
        cwd=ROOT,
    )

    (staging / "biophysj.tex").write_text(flattened_main_tex(), encoding="utf-8")

    for filename in [
        "biophys-new.cls",
        "biophysj.bst",
        "citation_biophysj.bib",
    ]:
        shutil.copy2(ROOT / filename, staging / filename)

    for filename in [
        "biophysj.bbl",
    ]:
        shutil.copy2(OUT / filename, staging / filename)

    (staging / "README.txt").write_text(
        "\n".join(
            [
                "AFM-Fold final clean manuscript source",
                "",
                "Manuscript ID: BIOPHYSICAL-JOURNAL-D-25-00813R1",
                "",
                "This flat package is intended for submission systems that reject",
                "LaTeX source trees containing subdirectories.",
                "",
                "Contents:",
                "- biophysj.tex: single-file LaTeX source with local inputs expanded",
                "- biophysj.pdf: clean final manuscript PDF without embedded figures",
                "- biophys-new.cls: Biophysical Journal class file",
                "- biophysj.bst: BibTeX style file required by Editorial Manager",
                "- citation_biophysj.bib: bibliography database",
                "- biophysj.bbl: generated bibliography for the manuscript",
                "- biophysj_SI.aux: cross-reference labels from the Supplemental Material",
                "",
                "Separate figure files and the composite Supplemental Material PDF",
                "should be uploaded separately, as requested by the journal.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_pdf(staging: Path) -> None:
    run(["pdflatex", "-interaction=nonstopmode", "biophysj"], cwd=staging)
    run(["biber", "biophysj"], cwd=staging)
    run(["pdflatex", "-interaction=nonstopmode", "biophysj"], cwd=staging)
    run(["pdflatex", "-interaction=nonstopmode", "biophysj"], cwd=staging)


def create_zip(staging: Path, zip_path: Path) -> None:
    names = [
        "README.txt",
        "biophysj.tex",
        "biophysj.pdf",
        "biophys-new.cls",
        "biophysj.bst",
        "citation_biophysj.bib",
        "biophysj.bbl",
        "biophysj_SI.aux",
    ]
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(staging / name, arcname=name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    staging = OUT / "_manuscript_no_figures_single_tex_build"
    prepare_staging(staging)
    build_pdf(staging)
    zip_path = OUT / ZIP_NAME
    create_zip(staging, zip_path)
    shutil.rmtree(staging)
    print(f"wrote {zip_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
