from fastapi import FastAPI
from contextlib import asynccontextmanager
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Startup...")
    time.sleep(10)
    print("Done startup")
    yield

app = FastAPI(lifespan=lifespan)
