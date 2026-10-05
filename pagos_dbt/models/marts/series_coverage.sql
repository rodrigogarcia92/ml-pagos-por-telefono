-- Coverage audit: one row per series, answering "what does this series actually
-- cover, and are there holes in it?"
--
-- WHY THIS IS A WAREHOUSE MODEL AND kappa IS NOT
--
-- Coverage is a FACT computed from the data: BCRP either published a month or
-- it did not, and no human opinion enters. Publication lag (kappa) is an
-- ASSUMPTION a human declared and can be wrong about -- it lives in
-- src/data_collection/config.py, versioned with the protocol that acts on it,
-- and reaches the modelling code through the snapshot rather than through
-- BigQuery. Facts here; assumptions there. See docs/training_plan.md 4.1.
--
-- TWO KINDS OF NULL, AND ONLY ONE IS A BUG
--
-- monthly_panel pivots 25 series with four different start dates into one wide
-- table, so a 2019 row has NULL for every wallet column. Those EDGE nulls are
-- correct: BCRP published no Yape/Plin split before 2024-01, and Plin did not
-- exist until 2020. INTERIOR nulls -- a hole inside a series' own coverage --
-- would be a real defect: a failed request, a dropped row, or a genuine BCRP
-- gap, any of which would silently poison a lag or a moving average.
--
-- Nothing distinguishes the two by eye in the wide panel. This model does, and
-- the not-null/accepted-range tests on interior_gaps turn a query someone ran
-- once into an assertion that runs on every dbt build.
--
-- Audited 2026-09-05: interior_gaps = 0 on all 25 series.

{{ config(materialized='view') }}

with bounds as (

    select
        meta.col_name,
        meta.category,
        meta.series_code,
        min(obs.obs_date)   as first_obs,
        max(obs.obs_date)   as last_obs,
        count(obs.value)    as n_values

    from {{ ref('stg_bcrp_observations') }} as obs

    inner join {{ ref('stg_series_metadata') }} as meta
        on meta.series_code = obs.series_code

    group by 1, 2, 3

)

select
    col_name,
    category,
    series_code,
    first_obs,
    last_obs,
    n_values,

    -- Months spanned, inclusive of both endpoints. All series are monthly
    -- (config.py enforces `frequency`), so for a complete series this equals
    -- n_values exactly.
    date_diff(last_obs, first_obs, month) + 1 as span_months,

    -- The number that matters. Zero means the series is dense between its own
    -- endpoints; anything else is a hole and must be explained before the
    -- series is used in a feature.
    date_diff(last_obs, first_obs, month) + 1 - n_values as interior_gaps,

    -- How stale this series is relative to the freshest series in the panel.
    -- This is the RAGGED RIGHT EDGE, and it is the empirical shadow of the
    -- publication lags in training_plan.md 4.1: at any pull date the policy
    -- rate is current, monthly macro is a month behind, and monthly GDP and
    -- the payments series are further behind still.
    --
    -- It is a cross-check on kappa, not a replacement for it: the panel's edge
    -- also depends on WHEN the ETL last ran, whereas kappa is about when BCRP
    -- publishes. Read a disagreement as a question, not an answer.
    date_diff(
        (select max(obs_date) from {{ ref('stg_bcrp_observations') }}),
        last_obs,
        month
    ) as months_behind_panel_edge

from bounds
order by category, col_name
