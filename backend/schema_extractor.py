"""
schema_extractor.py
────────────────────────────────────────────────────────────────────────────────
Connects to AdventureWorks2019 on MSSQL and extracts:
  • All tables (with the schemas listed in config.TARGET_SCHEMAS)
  • All columns per table  — name, data_type, nullable
  • Primary key columns per table
  • Foreign key relationships  — (child_table.child_col → parent_table.parent_col)

Output is saved to schema.json in the format:
{
  "tables": {
    "Sales.SalesOrderHeader": {
      "columns": [
        {"name": "SalesOrderID", "data_type": "int", "nullable": false},
        ...
      ],
      "primary_keys": ["SalesOrderID"],
      "foreign_keys": [
        {
          "column":           "CustomerID",
          "referenced_table": "Sales.Customer",
          "referenced_column":"CustomerID"
        },
        ...
      ]
    },
    ...
  },
  "foreign_key_relationships": [
    {
      "child_table":        "Sales.SalesOrderHeader",
      "child_column":       "CustomerID",
      "parent_table":       "Sales.Customer",
      "parent_column":      "CustomerID",
      "constraint_name":    "FK_SalesOrderHeader_Customer_CustomerID"
    },
    ...
  ]
}

Run:
    python schema_extractor.py
"""

import json
import sys
from textwrap import indent

from sqlalchemy import create_engine, text

from .config import CONNECTION_STRING, SCHEMA_JSON, TARGET_SCHEMAS


# ── SQL Queries ───────────────────────────────────────────────────────────────

_COLUMNS_SQL = """
SELECT
    t.TABLE_SCHEMA      AS schema_name,
    t.TABLE_NAME        AS table_name,
    c.COLUMN_NAME       AS column_name,
    c.DATA_TYPE         AS data_type,
    c.IS_NULLABLE       AS is_nullable
FROM
    INFORMATION_SCHEMA.TABLES   t
    JOIN INFORMATION_SCHEMA.COLUMNS c
        ON  c.TABLE_SCHEMA = t.TABLE_SCHEMA
        AND c.TABLE_NAME   = t.TABLE_NAME
WHERE
    t.TABLE_TYPE   = 'BASE TABLE'
    AND t.TABLE_SCHEMA IN :schemas
ORDER BY
    t.TABLE_SCHEMA, t.TABLE_NAME, c.ORDINAL_POSITION
"""

_PK_SQL = """
SELECT
    tc.TABLE_SCHEMA,
    tc.TABLE_NAME,
    ku.COLUMN_NAME
FROM
    INFORMATION_SCHEMA.TABLE_CONSTRAINTS   tc
    JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
        ON  ku.CONSTRAINT_NAME   = tc.CONSTRAINT_NAME
        AND ku.TABLE_SCHEMA      = tc.TABLE_SCHEMA
        AND ku.TABLE_NAME        = tc.TABLE_NAME
WHERE
    tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
    AND tc.TABLE_SCHEMA IN :schemas
ORDER BY
    tc.TABLE_SCHEMA, tc.TABLE_NAME, ku.ORDINAL_POSITION
"""

_FK_SQL = """
SELECT
    fk.name                                              AS constraint_name,
    SCHEMA_NAME(parent_obj.schema_id)                    AS child_schema,
    parent_obj.name                                      AS child_table,
    parent_col.name                                      AS child_column,
    SCHEMA_NAME(ref_obj.schema_id)                       AS parent_schema,
    ref_obj.name                                         AS parent_table,
    ref_col.name                                         AS parent_column
FROM
    sys.foreign_keys                    fk
    JOIN sys.foreign_key_columns        fkc  ON fkc.constraint_object_id = fk.object_id
    JOIN sys.objects                    parent_obj  ON parent_obj.object_id  = fk.parent_object_id
    JOIN sys.columns                    parent_col  ON parent_col.object_id  = fk.parent_object_id
                                                   AND parent_col.column_id  = fkc.parent_column_id
    JOIN sys.objects                    ref_obj     ON ref_obj.object_id     = fk.referenced_object_id
    JOIN sys.columns                    ref_col     ON ref_col.object_id     = fk.referenced_object_id
                                                   AND ref_col.column_id     = fkc.referenced_column_id
WHERE
    SCHEMA_NAME(parent_obj.schema_id) IN :schemas
ORDER BY
    child_schema, child_table, constraint_name
"""


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_columns(conn, schemas: tuple) -> dict:
    """
    Returns:
        { "Schema.Table": [{"name":..., "data_type":..., "nullable":...}, ...] }
    """
    rows = conn.execute(text(_COLUMNS_SQL), {"schemas": schemas}).fetchall()
    table_columns: dict[str, list] = {}
    for row in rows:
        full = f"{row.schema_name}.{row.table_name}"
        table_columns.setdefault(full, []).append({
            "name":      row.column_name,
            "data_type": row.data_type,
            "nullable":  row.is_nullable == "YES",
        })
    return table_columns


def _extract_primary_keys(conn, schemas: tuple) -> dict:
    """
    Returns:
        { "Schema.Table": ["col1", "col2", ...] }
    """
    rows = conn.execute(text(_PK_SQL), {"schemas": schemas}).fetchall()
    pk_map: dict[str, list] = {}
    for row in rows:
        full = f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}"
        pk_map.setdefault(full, []).append(row.COLUMN_NAME)
    return pk_map


def _extract_foreign_keys(conn, schemas: tuple) -> list:
    """
    Returns a flat list of FK relationship dicts.
    """
    rows = conn.execute(text(_FK_SQL), {"schemas": schemas}).fetchall()
    fk_list = []
    for row in rows:
        child_full  = f"{row.child_schema}.{row.child_table}"
        parent_full = f"{row.parent_schema}.{row.parent_table}"
        fk_list.append({
            "constraint_name":  row.constraint_name,
            "child_table":      child_full,
            "child_column":     row.child_column,
            "parent_table":     parent_full,
            "parent_column":    row.parent_column,
        })
    return fk_list


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_schema(connection_string: str, target_schemas: list[str]) -> dict:
    """
    Connects to MSSQL, runs all three queries, and assembles the schema dict.
    """
    schemas_tuple = tuple(target_schemas)

    print(f"Connecting to database …")
    engine = create_engine(connection_string, fast_executemany=True)

    with engine.connect() as conn:
        print("  ▸ Extracting columns …")
        table_columns = _extract_columns(conn, schemas_tuple)

        print("  ▸ Extracting primary keys …")
        pk_map = _extract_primary_keys(conn, schemas_tuple)

        print("  ▸ Extracting foreign keys …")
        fk_list = _extract_foreign_keys(conn, schemas_tuple)

    # ── Attach PKs and per-table FKs ─────────────────────────────────────────
    # Build per-table FK lookup (child side)
    table_fks: dict[str, list] = {}
    for fk in fk_list:
        table_fks.setdefault(fk["child_table"], []).append({
            "column":            fk["child_column"],
            "referenced_table":  fk["parent_table"],
            "referenced_column": fk["parent_column"],
        })

    tables_dict: dict[str, dict] = {}
    for full_table, cols in sorted(table_columns.items()):
        tables_dict[full_table] = {
            "columns":      cols,
            "primary_keys": pk_map.get(full_table, []),
            "foreign_keys": table_fks.get(full_table, []),
        }

    return {
        "tables":                    tables_dict,
        "foreign_key_relationships": fk_list,
    }


# ── Serialise ─────────────────────────────────────────────────────────────────

def save_schema(schema: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"\nSchema saved → {path}")


# ── Pretty summary ────────────────────────────────────────────────────────────

def print_summary(schema: dict) -> None:
    n_tables  = len(schema["tables"])
    n_cols    = sum(len(v["columns"]) for v in schema["tables"].values())
    n_fks     = len(schema["foreign_key_relationships"])

    print("\n" + "=" * 60)
    print(f"  Tables   : {n_tables}")
    print(f"  Columns  : {n_cols}")
    print(f"  FK edges : {n_fks}")
    print("=" * 60)

    print("\nSample (first 5 tables):")
    for i, (tbl, meta) in enumerate(schema["tables"].items()):
        if i >= 5:
            break
        pk_str  = ", ".join(meta["primary_keys"]) or "—"
        fk_cnt  = len(meta["foreign_keys"])
        col_cnt = len(meta["columns"])
        print(f"  {tbl:<45} cols={col_cnt:<4} pk=[{pk_str}]  fks={fk_cnt}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Agentic Hybrid RAG — Schema Extractor")
    print("=" * 60)
    print(f"Target schemas : {TARGET_SCHEMAS}\n")

    try:
        schema = extract_schema(CONNECTION_STRING, TARGET_SCHEMAS)
    except Exception as exc:
        print(f"\n❌  Connection failed: {exc}", file=sys.stderr)
        print("\nTroubleshooting tips:")
        print("  • Make sure SQL Server is running and the server name in config.py is correct.")
        print("  • Ensure ODBC Driver 17 for SQL Server is installed.")
        print("  • Verify that AdventureWorks2019 is attached/restored.")
        sys.exit(1)

    print_summary(schema)
    save_schema(schema, SCHEMA_JSON)
    print("\n✅  Schema extraction complete.")


if __name__ == "__main__":
    main()
