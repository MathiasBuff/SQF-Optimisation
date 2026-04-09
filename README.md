# SQF Score Optimisation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MathiasBuff/SQF-Optimisation/HEAD?urlpath=%2Fdoc%2Ftree%2FNotebook.ipynb)

A research project on score-based chromatographic method optimisation from limited scouting data.

This repository explores a lightweight, model-based workflow for chromatographic method development using objective separation scores. Starting from a small set of scouting measurements, it fits simple retention and peak-width models, predicts chromatographic behaviour across a two-dimensional method space, computes separation-oriented subscores and a global SQF score, and identifies promising operating conditions.

The project is intended to help analytical chemists reduce empirical trial-and-error during early method development, while keeping the workflow accessible and transparent.

## Project scope

The current implementation focuses on:

- fitting predictive models from a limited experimental dataset,
- exploring a 2D method space defined by temperature and gradient time,
- comparing operating conditions through score-based criteria,
- visualising predicted score surfaces,
- identifying an optimal predicted condition,
- simulating a synthetic chromatogram at selected conditions.

At this stage, the project should be seen as a research and prototyping tool rather than a finished end-user application.

## Current interface

The main user-facing entry point is the Jupyter notebook. It serves as the default wrapper around the underlying Python package and provides a straightforward top-to-bottom workflow for standard use.

The computational logic is implemented in the `sqf_optimisation` package, allowing more advanced users to reuse the underlying modules directly in their own scripts or notebooks.

A standalone graphical interface may be developed later, but this is not the current focus of the project.

## Repository structure

```text
SQF-Optimisation/
├── data/                 # Example input files
├── sqf_optimisation/     # Core package
├── Notebook.ipynb        # Main notebook interface
├── requirements.txt
└── README.md
```

## Access

The notebook can be launched directly through Binder using the badge above.

Alternatively, the repository can be cloned and run locally in a standard Python/Jupyter environment.

## Input data

The workflow currently expects:

* a configuration file describing the instrument and method domain,
* a measurements file containing the scouting experimental data.

For now, the expected file structure is best understood from the examples in the `data/` folder and from the notebook itself. More formal input documentation may be added later.

## Status

This project is an active work in progress. The core workflow is already usable, but the documentation and presentation are still being refined, and the overall tool is still evolving.

## License

MIT License.
