# CBLUE source record

- Official project: https://github.com/CBLUEbenchmark/CBLUE
- Official data page: https://tianchi.aliyun.com/dataset/95414
- Official-code commit: `6a2c54f6a69265a33181c752a5350277149b8883`
- Public data mirror: https://github.com/Sherlock-coder/CBLUE
- Mirror commit: `8f879e3624b06a184288a8770113da5fddcb61ff`
- Download date: 2026-09-15

## Local contents

- `official_code/`: shallow clone of the official CBLUE implementation.
- `cblue1_mirror/`: public mirror containing the eight CBLUE1 task datasets.

The official Tianchi data package requires an authenticated account and dataset
application. Therefore, the data currently present are from the named public
mirror and must not be confused with a fresh export from Tianchi. Confirm the
official dataset terms before redistribution or publication.

All JSON/JSONL files under `cblue1_mirror/data/` passed parsing checks. The
verified training-set sizes are CMeEE 15,000; CMeIE 14,339; CHIP-CDN 6,000;
CHIP-CTC 22,962; CHIP-STS 16,000; KUAKE-QIC 6,931; KUAKE-QTR 24,174; and
KUAKE-QQR 15,000. The mirror itself contains a duplicated nested CHIP-CTC copy;
future loaders should target only the first-level task files under `data/`.
