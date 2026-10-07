-- Google Trends, collapsed to ONE value per month from ONE pull.
--
-- raw.trends_observations is append-only and holds every pull (and, per pull, both
-- entities as exported: Yape "Aplicación" and Plin "Tema"). This model does the two
-- things nothing downstream may be left to remember (training_plan.md 4.5):
--
--   1. THE SINGLE-PULL RULE.  Trends rescales every request to its own peak (0-100),
--      so months from two pulls are NOT on one scale. A data_version therefore takes
--      the WHOLE history from ONE pull -- never month-by-month de-duplication, which is
--      what stg_bcrp_observations does and what would be wrong here. Which pull is a
--      declared choice, recorded in dbt_project.yml as var `trends_pull_id`: the latest
--      pull that passed the data gate (docs/training_plan.md 9.2). If the var is unset
--      the newest pull wins, which is a convenience and not a protocol decision.
--
--   2. THE CONSOLIDATED SERIES.  yape + plin, in index points, from one request. A
--      month with a missing entity is NULL rather than a half-sum.
--
-- The singular tests assert_stg_trends_single_pull and assert_panel_gt_from_one_pull
-- are what make rule 1 a check and not a comment.

{{ config(enabled=var('enable_trends', false)) }}

{%- set chosen = var('trends_pull_id', none) %}

with pulls as (

    select * from {{ source('raw', 'trends_observations') }}

    {%- if chosen %}
    where pull_id = '{{ chosen }}'
    {%- else %}
    where pull_id = (select max(pull_id) from {{ source('raw', 'trends_observations') }})
    {%- endif %}

)

select
    pulls.series_code,
    pulls.obs_date,
    if(count(pulls.value) = 2 and count(distinct pulls.entity) = 2,
       sum(pulls.value), null)                       as value,
    pulls.pull_id,
    min(pulls.pull_date)                             as pull_date,
    min(pulls.source_batch)                          as source_batch

from pulls

inner join {{ ref('stg_series_metadata') }} as meta
    on meta.series_code = pulls.series_code

group by pulls.series_code, pulls.obs_date, pulls.pull_id
