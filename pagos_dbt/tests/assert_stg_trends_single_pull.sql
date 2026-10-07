{{ config(enabled=var('enable_trends', false)) }}

-- T8 (training_plan.md 9.2), part 1. The single-pull rule: stg_trends must hold
-- EXACTLY ONE pull_id. Returns a row (= the test fails) when it holds none -- the
-- var `trends_pull_id` names a pull that is not in raw -- or more than one.
--
-- One pull_date per data_version: the snapshot takes its Trends version from this
-- single pull (src/model_training/snapshot.py), so two pulls here would be two
-- scales inside one data_version.

select
    count(distinct pull_id) as n_pulls,
    string_agg(distinct pull_id) as pull_ids
from {{ ref('stg_trends') }}
having count(distinct pull_id) != 1
