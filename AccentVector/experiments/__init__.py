"""Unfinished analyses — written, never run to completion, kept for the write-up.

This package is NOT part of the working pipeline, which is why it sits outside the
installable ``src/accent_vector`` package. Run a module from ``AccentVector/`` with the
package on the path:

    PYTHONPATH=src python -m experiments.<module> ...

Everything the dissertation actually depends on lives in ``src/accent_vector/``:

    data_preprocess -> (finetune on Eddie) -> extract_vector -> infer_accent
                    -> score_sweep / score_prosody -> ../notebooks/dissertation_figures.ipynb

The modules below were written against the experiment matrix
but produced no output in this checkout: none of ``temporal.csv``,
``by_step_summary.csv``, ``rq3_layers.csv`` or ``gram.csv`` exists anywhere under AccentVector/. They are
kept because the questions are still open, not because they are wired up.

    rq2_behavioural.py  output metrics at matched alpha across checkpoints. Its
                        inputs (per-step rq1.csv) now EXIST, so this is the one
                        module here that could run today -- the notebook's Ch.4
                        trajectory section answers the same question by a
                        different route (local slope + settling step).
    rq2_temporal.py     weight-space trajectory: ||tau_t||, direction convergence
    viz_temporal.py     dashboard video of that trajectory (needs rq2_temporal)
    rq3_layers.py       which modules/depths carry the accent shift
    rq5_gram.py         accent-vector Gram matrix: cosines, eigenspectrum, LOO
    rq5_walkthrough.py  narrated tutorial over rq5_gram's objects

Note the RQ numbers in these filenames predate the current chapter structure:
what they call RQ2 the dissertation now calls RQ1b, and no RQ5 section exists.
Rename them if any of them graduates into the pipeline.

Removed rather than kept: ``evaluate.py`` (superseded by score_sweep / score_prosody),
``aggregate.py`` (the notebook pools speakers itself), ``rq5_geometry.py`` (its
weight-space cosines used raw LoRA factors, superseded by rq5_gram's dW = B@A; its
output-space GenAID matrix + Mantel test had no replacement), and ``grid.py`` and
``checkpoint_grid.py``, which built the
synthesis grid on a GPU. The Eddie shell scripts do that job now
(``submit_*_ckpt_grid.sh`` -> ``eddie_infer_array.sh`` -> ``infer_sweep.sh`` ->
``accent_vector.infer_accent``), and no ``grid.json`` was ever written. They are
in git history if the standalone drivers are ever wanted back.

Plotting lives in ``../notebooks/dissertation_figures.ipynb``.
"""
