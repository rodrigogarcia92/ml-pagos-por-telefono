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

{%- set enable_trends = var('enable_trends', false) -%}
{%- set trends_cols = var('trends_col_names', ['gt_yape_plin']) -%}

{%- set all_col_names = dbt_utils.get_column_values(
        table=source('raw', 'series_metadata'),
        column='col_name',
        order_by='col_name',
        default=[]
) -%}
{#- Trends is opt-in (var enable_trends): off, its column is absent, not a column of NULLs. -#}
{%- set col_names = all_col_names if enable_trends else all_col_names | reject('in', trends_cols) | list -%}

select
    obs.obs_date

    {%- for col in col_names %},
    max(if(meta.col_name = '{{ col }}', obs.value, null)) as {{ col }}
    {%- endfor %}

from (

    -- BCRP: latest pull per (series, month). Trends: ONE pull, whole history (stg_trends).
    -- Different de-duplication rules on purpose; see stg_trends.sql.
    select series_code, obs_date, value from {{ ref('stg_bcrp_observations') }}
    {%- if enable_trends %}
    union all
    select series_code, obs_date, value from {{ ref('stg_trends') }}
    {%- endif %}

) as obs

inner join {{ ref('stg_series_metadata') }} as meta
    on meta.series_code = obs.series_code

group by obs.obs_date
order by obs.obs_date
