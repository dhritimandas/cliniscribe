"""Central configuration: tunables shared by the pipeline and the eval harness.

Every value here is a measured decision, not a default — see LEARNINGS.md for
the study behind each. Eval and production MUST read the same values so that
benchmark numbers describe the system as shipped.
"""

# L3 ASR decoding beam width. Reconciled by eval/beam_study.py on the frozen
# 10-clip Hindi bench (drug/keyword WER, token-boundary scorer) — see
# outputs/beam_study.json for the measurement behind the pinned value.
ASR_BEAM_SIZE = 5
