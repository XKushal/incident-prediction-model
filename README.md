# ResolveAI

Initial project structure for working with the UCI incident event log dataset.
Dataset preparation has not been implemented yet.

## Project structure

```text
ResolveAI/
├── data/
│   └── incident_event_log.csv
├── src/
│   └── prepare_dataset.py
├── README.md
└── .gitignore
```

## Original data

Place the original UCI CSV at `data/incident_event_log.csv`. Keep this file
unchanged; future preparation code must write derived data to a separate file.
The local copy was copied from the existing downloaded file and verified
byte-for-byte using SHA-256:

```text
fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef
```

The dataset is intentionally excluded from Git until its source attribution
and distribution approach are documented. Virtual environments and Python
cache files are also ignored.

## Next step

Implement and verify `src/prepare_dataset.py`, then make the initial push
to GitHub. The current file is a documented placeholder.
