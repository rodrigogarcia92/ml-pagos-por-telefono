"""Modelling package — roadmap steps 7-10.

Training is local by design (see docs/methodology.md). No module here
queries BigQuery: everything reads the parquet snapshot written by snapshot.py
(plan §4.0).
"""
