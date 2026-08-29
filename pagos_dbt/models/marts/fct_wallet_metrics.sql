-- Derived Yape/Plin measures, defined once here rather than re-typed in each
-- notebook. Replaces the ad-hoc n_transf_total / v_transf_digital / yape-share
-- cells at the bottom of notebooks/01_eda_bcrp.py.
--
-- Restricted to months where the target series exist. monthly_panel now spans
-- 2010 onwards (macro series carry decades of history), but the Yape/Plin
-- breakdown only begins in Jan 2024 — so most rows would otherwise be all-null
-- here. The filter is on the count series specifically, since that is what the
-- wallet split is keyed on.

{{ config(materialized='table') }}

with wallets as (

    select
        obs_date,
        n_transf_intra_yape + n_transf_inter_visa_yape as n_yape,
        n_transf_intra_plin + n_transf_inter_visa_plin as n_plin,
        v_transf_intra_yape + v_transf_inter_visa_yape as v_yape,
        v_transf_intra_plin + v_transf_inter_visa_plin as v_plin

    from {{ ref('monthly_panel') }}
    where n_transf_intra_yape is not null

),

totals as (

    select
        *,
        n_yape + n_plin as n_total,
        v_yape + v_plin as v_total
    from wallets

)

select
    obs_date,

    n_yape,
    n_plin,
    n_total,
    v_yape,
    v_plin,
    v_total,

    -- safe_divide returns NULL instead of erroring on a zero denominator.
    safe_divide(n_yape, n_total) as yape_share_numero,
    safe_divide(v_yape, v_total) as yape_share_valor,

    -- Average ticket. UNITS UNVERIFIED: BCRP publishes counts and amounts on
    -- different scales (miles vs millones), so this ratio is directionally
    -- right but not yet in soles. Confirm the scaling factors against the API
    -- metadata before quoting this figure anywhere.
    safe_divide(v_total, n_total) as ticket_promedio_raw,

    -- Month-over-month growth. LAG over an ordered window; the first row is
    -- NULL by construction, which is correct — there is no prior month.
    safe_divide(n_total, lag(n_total) over (order by obs_date)) - 1 as n_total_mom,
    safe_divide(v_total, lag(v_total) over (order by obs_date)) - 1 as v_total_mom

from totals
order by obs_date
