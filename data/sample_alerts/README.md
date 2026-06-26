# Sample alerts fixture

This folder is for testing `scripts/bulk_ingest.py`.

It follows the same camera-folder layout that the script expects:

```text
data/sample_alerts/
├── cam00/
│   ├── image_001.png
│   ├── image_002.png
│   └── metadata.json
├── cam01/
│   ├── image_001.png
│   └── metadata.json
└── cam02/
    ├── image_001.png
    └── metadata.json
```

The PNGs are tiny valid image files that let you test the upload path end to end.
The `metadata.json` files are sample sidecar data you can later teach the API to read when you extend ingest.