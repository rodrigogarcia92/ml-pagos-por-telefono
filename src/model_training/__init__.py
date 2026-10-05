"""Modelling package — roadmap steps 7-10.

Training is local by design (docs/methodology.md §8.1). No module here
queries BigQuery: everything reads the parquet snapshot written by snapshot.py
(docs/training_plan.md §4.0).
"""
