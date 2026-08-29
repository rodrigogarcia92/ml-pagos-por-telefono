-- The single most important model in this project.
--
-- raw.bcrp_observations is append-only and currently holds two pull dates, so
-- every (series, month) can appear more than once. Any query that forgets to
-- collapse those duplicates silently double-counts — in a feature matrix, in a
-- chart, in a training set.
--
-- This model makes that impossible to get wrong: nothing downstream reads the
-- raw table, so nothing downstream has to remember the rule.
--
-- ROW_NUMBER, not RANK: snapshots are stamped by date, so two pulls on the
-- same day would tie. RANK would return both rows and reintroduce the very
-- duplication this model exists to remove. ROW_NUMBER always yields exactly
-- one row per (series, month).

select
    series_code,
    obs_date,
    value,
    pulled_at,
    source_batch

from {{ source('raw', 'bcrp_observations') }}

qualify row_number() over (
    partition by series_code, obs_date
    order by pulled_at desc
) = 1
