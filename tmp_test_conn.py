import sqlalchemy

cs = "mssql+pymssql://sqladmin:12345%40Hv@rag-sql-server.database.windows.net/AdventureWorks2019"
engine = sqlalchemy.create_engine(cs, connect_args={"timeout": 30})

with engine.connect() as conn:
    db = conn.execute(sqlalchemy.text("SELECT DB_NAME()")).fetchone()
    print(f"Connected to DB: {db[0]}")

    # Check all object types in the DB
    objs = conn.execute(sqlalchemy.text(
        "SELECT type_desc, COUNT(*) as cnt FROM sys.objects GROUP BY type_desc ORDER BY cnt DESC"
    )).fetchall()
    print("\nAll objects in DB:")
    for o in objs:
        print(f"  {o[0]}: {o[1]}")

    # Check all schemas
    schemas = conn.execute(sqlalchemy.text(
        "SELECT name FROM sys.schemas"
    )).fetchall()
    print(f"\nSchemas: {[s[0] for s in schemas]}")

    # Check import history / operations
    state = conn.execute(sqlalchemy.text(
        "SELECT state_desc FROM sys.databases WHERE name = DB_NAME()"
    )).fetchone()
    print(f"\nDB state: {state[0] if state else 'unknown'}")
