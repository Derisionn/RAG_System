class IngestController:
    def trigger_ingest(self) -> dict:
        """Trigger ingestion pipelines."""
        return {"status": "accepted", "detail": "Ingestion job stub completed."}
