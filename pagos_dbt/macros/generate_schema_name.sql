{#
    Override dbt's default schema naming.

    Out of the box, dbt CONCATENATES the target schema with the custom one:
    target `staging` + `+schema: marts` produces the dataset `staging_marts`.
    That default exists so several developers can build into the same warehouse
    without colliding — each gets their own prefix.

    This project has one developer and a warehouse layout already documented in
    docs/data_sources.md (`raw` / `staging` / `marts`). Honouring `+schema`
    verbatim keeps the datasets matching the design.

    Trade-off, stated plainly: this gives up dbt's built-in environment
    separation. If a second target (e.g. `prod`) is ever added, dev and prod
    would write to the SAME datasets. The fix at that point is to prefix by
    target name here rather than to delete this macro.
#}

{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
