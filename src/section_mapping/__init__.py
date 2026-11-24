"""Mapping partial 2-D tissue sections onto a 3-D labelled atlas (dataset-independent methods).

Modules, in the order of the methodology (docs/section_mapping.md):

``measures``       representation: particle measures, kernel inner products, compression
``atlas``          anatomical localization: label tables, feature channels, planes and
                   reslicing, region support, landing, landmark-slab prior
``alignment``      geometric alignment (CPU reference): 2-D similarity, support-censored
                   varifold score, FFT search + refinement, pose algebra, common post-fit score
``alignment_gpu``  the same score, search and refinement batched in torch
``stacking``       roles QC, depth profiles, noise scale, windows, escalation, decision rules,
                   placement diagnostics

Deformable registration is xIV-LDDMM (vendored), driven by
``src/scripts/framework_script_AtlasToSectionCrossModalBoundary.py``.  Dataset-specific
workflows (SEA-AD QNP: ``src/scripts/map_qnp_sections.py``) read files, pick sections and call
these functions; nothing here reads a project config or knows a donor.
"""
