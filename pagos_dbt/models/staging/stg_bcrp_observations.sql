-- The single most important model in this project.
--
-- raw.bcrp_observations is append-only and now holds several pull dates, so
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
--
-- ---------------------------------------------------------------------------
-- RETIRED SERIES — added 2026-09-05
--
-- The second job of this model: present only the series config.py currently
-- declares.
--
-- raw.bcrp_observations is an IMMUTABLE ARCHIVE. raw.series_metadata is a
-- CURRENT DECLARATION, rewritten from config.py on every loader run. Those two
-- facts are both deliberate, and together they mean that retiring a series
-- leaves its observations behind in raw forever, with no metadata row to join
-- to. Dropping three CCE series left exactly 93 orphan rows — 3 series x 31
-- months — and the referential-integrity test caught it on the next build.
--
-- Three ways to resolve that, and only one is right:
--
--   * Delete the orphans from raw. Destroys the archive, and the archive is
--     what makes the warehouse rebuildable from data/raw/bcrp/*.json.
--   * Relax the test to a warning. Hides the class of bug the test exists for.
--   * Filter here. The staging layer is where "raw history" becomes "the
--     current curated view", which is precisely this decision.
--
-- So the join below is an INNER join and it is load-bearing, not cosmetic.
-- Re-adding a retired series is still one line in config.py plus a loader run:
-- its history never left.
--
-- Consequence for the test suite: the relationships test on series_code now
-- passes by construction and is kept only as a regression guard. The test that
-- carries real information after this change is the one in the OTHER
-- direction — every declared series must actually have observations — which
-- catches the realistic failure, a typo'd BCRP code in config.py.
-- ---------------------------------------------------------------------------

select
    obs.series_code,
    obs.obs_date,
    obs.value,
    obs.pulled_at,
    obs.source_batch

from {{ source('raw', 'bcrp_observations') }} as obs

inner join {{ ref('stg_series_metadata') }} as meta
    on meta.series_code = obs.series_code

qualify row_number() over (
    partition by obs.series_code, obs.obs_date
    order by obs.pulled_at desc
) = 1
