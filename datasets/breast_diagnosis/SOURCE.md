# BREAST-DIAGNOSIS source record

- Source: The Cancer Imaging Archive / NCI Imaging Data Commons
- Collection ID: `breast_diagnosis`
- DOI: https://doi.org/10.7937/K9/TCIA.2015.SDNRQXXR
- License: CC BY 3.0
- Download tool: `idc-index` 0.12.5
- Download date: 2026-09-15

## Local contents

- `raw/`: DICOM files downloaded with `idc download breast_diagnosis`.
- `clinical/`: IDC clinical table exported in CSV and Parquet formats.
- `metadata/`: 523-series IDC manifest in CSV and Parquet formats.
- `metadata/download_audit.json`: post-download series and instance audit.
- `download.log`: transfer log.

IDC reports 60,871.10 MB and 105,144 instances across 523 series: MR, MG, CT,
PT, and SR. This is a
breast-cancer imaging collection, dominated by MRI, and contains no ultrasound.
It is not a mastitis-specific dataset. The clinical export contains 51 records
and 18 fields, including pathology diagnosis, receptor status, Ki67, Oncotype,
BI-RADS, MRI impression, and pathology-report notes. It is the standardized IDC
version of supporting clinical data; the historical 21.72 KB TCIA XLSX is not
part of the DICOM download.

Post-download verification matched all 523 expected series and all 105,144
expected instances (60,871,104,492 bytes), with no zero-byte files and no
series-count mismatches.
