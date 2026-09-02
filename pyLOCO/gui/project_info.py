"""Canonical pyLOCO Suite project metadata shared by all applications."""

PROJECT_REPOSITORY = "https://github.com/elafmusa/pyLOCO"
PROJECT_DOCUMENTATION = f"{PROJECT_REPOSITORY}#readme"
PROJECT_PAPER_URL = "https://indico.jacow.org/event/95/contributions/13338/"
PROJECT_ISSUES = f"{PROJECT_REPOSITORY}/issues"
PROJECT_LICENSE = "Apache-2.0"
PROJECT_PAPER_TITLE = "PyLOCO: A Python Framework for Linear Optics Correction in Storage Rings"
PROJECT_CONTRIBUTORS = "Elaf Musa"
PROJECT_ACKNOWLEDGEMENTS = "Ilya Agapov, Joachim Keil, Konstantinos Paraschou, Simone Liuzzo, and Ahmed El Deeb"


def citation_text() -> str:
    return f"E. Musa, I. Agapov, K. Paraschou, J. Keil, and S. Liuzzo, ‘{PROJECT_PAPER_TITLE},’ presented at IPAC’26, Deauville, France, May 2026, paper WEP5011. {PROJECT_PAPER_URL}"


def bibtex_text() -> str:
    return "\n".join(("@inproceedings{musa_pyloco_ipac26,", "  author = {Musa, Elaf and Agapov, Ilya and Paraschou, Konstantinos and Keil, Joachim and Liuzzo, Simone},", f"  title = {{{PROJECT_PAPER_TITLE}}},", "  booktitle = {Proceedings of the 17th International Particle Accelerator Conference (IPAC'26)},", "  year = {2026},", "  note = {Paper WEP5011},", f"  url = {{{PROJECT_PAPER_URL}}}", "}"))
