# Aquarius Artifacts

This directory contains Project Aquarius metadata and verification artifacts.

## Structure

- `results/` - Experiment results and benchmark outputs (JSON files following the results artifact schema)

## Adding Results

Create a JSON file in `results/` following this schema:

```json
{
  "schema_version": "1.0.0",
  "result_id": "unique-experiment-id",
  "title": "Experiment Title",
  "updated_at": "2025-12-30T12:00:00Z",
  "summary": "Brief description of results",
  "metrics": {
    "your_metric": 42.0,
    "paper_metric": 43.0,
    "delta_pct": 2.3
  }
}
```

See the org schema docs at: https://github.com/Project-Aquarius-White/.github/tree/main/schemas
