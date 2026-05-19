import sys
import urllib.parse
from sqlalchemy import create_engine, text

# We will try both passwords:
passwords = ["12345", "12345@Hv", "8630466531harsh"]
# Try pooler first, then direct host
hosts = [
    ("aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres", "postgres.imtfgdgjqwjifljsuhgi"),
    ("db.imtfgdgjqwjifljsuhgi.supabase.co:5432/postgres", "postgres")
]

for host, user in hosts:
    for pwd in passwords:
        encoded_pwd = urllib.parse.quote_plus(pwd)
        conn_str = f"postgresql+psycopg2://{user}:{encoded_pwd}@{host}"
        print(f"Trying User={user}, Host={host}, Pwd={pwd}...")
        try:
            engine = create_engine(conn_str, connect_args={"connect_timeout": 5})
            with engine.connect() as conn:
                res = conn.execute(text("SELECT 1")).scalar()
                print(f"[SUCCESS] SELECT 1 returned: {res}")
                print(f"The correct connection string is: {conn_str}")
                sys.exit(0)
        except Exception as e:
            err_str = str(e)
            if "password authentication failed" in err_str:
                print(" -> Auth failed (Wrong password)")
            elif "timeout" in err_str or "timed out" in err_str:
                print(" -> Timeout (Network/ISP blocking)")
            else:
                print(f" -> Error: {err_str[:120]}")
print("Could not connect with any password.")
