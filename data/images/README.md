# Temporary image store fixture

This directory is a local-only placeholder for the API image store.

Layout:

- `cam-temp/` — dummy camera folder used for testing
- `cam-temp/metadata.json` — sample alert metadata
- `cam-temp/frame_001.png` and `cam-temp/frame_002.png` — tiny valid PNG placeholders

The backend currently saves uploads here via `IMAGE_STORE_DIR=./data/images`.
