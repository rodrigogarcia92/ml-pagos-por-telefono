-- Currently a pass-through: raw.series_metadata is rewritten wholesale from
-- config.py on every load, so there is nothing to clean or de-duplicate.
--
-- It exists anyway because of a convention worth keeping: downstream models
-- ref() staging models, never source() raw tables directly. That way, if the
-- raw table ever gains a column, changes a type, or needs a filter, there is
-- exactly one place to absorb the change — and every consumer inherits it.
-- The cost of an empty seam is nothing; the cost of not having one is a
-- find-and-replace across every mart.

select
    series_code,
    col_name,
    description,
    category,
    frequency

from {{ source('raw', 'series_metadata') }}
