-- Wide analysis table: one row per month, one column per series.
-- This is `df_wide` from the EDA notebook, rebuilt in SQL — so the feature
-- matrix is defined once, in the warehouse, instead of being re-derived in
-- every notebook that needs it.
--
-- The column list is NOT hardcoded. It is read at compile time from
-- raw.series_metadata, which the Python loader rewrites from config.py on
-- every run. So the chain is:
--
--     config.py  ->  raw.series_metadata  ->  these column names
--
-- Add a series to config.py, re-run the loader, re-run dbt, and the column
-- appears. No SQL edit, and no second list to keep in sync.
--
-- Why source() and not ref() for the column lookup: get_column_values runs
-- during COMPILATION, before any model is built. On a fresh warehouse the
-- staging view would not exist yet. raw.series_metadata always does — the
-- Python ETL created it.

{{ config(materialized='table') }}

{%- set col_names = dbt_utils.get_column_values(
        table=source('raw', 'series_metadata'),
        column='col_name',
        order_by='col_name',
        default=[]
) -%}

select
    obs.obs_date

    {%- for col in col_names %},
    max(if(meta.col_name = '{{ col }}', obs.value, null)) as {{ col }}
    {%- endfor %}

from {{ ref('stg_bcrp_observations') }} as obs

inner join {{ ref('stg_series_metadata') }} as meta
    on meta.series_code = obs.series_code

group by obs.obs_date
order by obs.obs_date
