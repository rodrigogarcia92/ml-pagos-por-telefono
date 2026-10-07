{{ config(enabled=var('enable_trends', false)) }}

-- T8 (training_plan.md 9.2), part 2. Every gt_yape_plin value in marts.monthly_panel
-- must be the value stg_trends holds for that month, i.e. come from the one pull.
-- Returns a row (= failure) for any month where the panel and stg_trends disagree,
-- including a panel month with a Trends value that stg_trends does not have and a
-- stg_trends month that never reached the panel.

with panel as (

    select obs_date, gt_yape_plin from {{ ref('monthly_panel') }}
    where gt_yape_plin is not null

),

staged as (

    select obs_date, value as gt_yape_plin from {{ ref('stg_trends') }}
    where value is not null

)

select
    coalesce(panel.obs_date, staged.obs_date) as obs_date,
    panel.gt_yape_plin  as in_panel,
    staged.gt_yape_plin as in_stg_trends
from panel
full outer join staged on panel.obs_date = staged.obs_date
where panel.gt_yape_plin is distinct from staged.gt_yape_plin
