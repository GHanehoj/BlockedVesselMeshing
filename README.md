# BlockedVesselMeshing

All dependencies listed in requirements.txt, though not all are needed for the core functionality..

Project structure:

* `data/`
    - `input/`: Currently contains 5 datasets from different sources, with each having the original data in the `raw/` subfolder, and the preprocessed (standardized) data in the `skeleton/`.
* `python/`: Source code. Most of the main functionality lies in the root, with `clusters.py` implementing the cluster splitting and `blocked.py` implementing the blocked approach.
    - `experiments/`: Various script files that implement different experiments or visualizations. Not all are up-to-date and runs.
    - `preprocessing/`: Code to convert input mesh formats into our standardized skeletons as well as the preprocessing pass which improves skeleton niceness.
    - `rainbow/`: A math library used as utility.
    - `tools/`: Various custom utilities and data formats.
