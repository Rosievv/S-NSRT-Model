# District-Level Validation: HS 9403 Furniture x Panama Canal Drought 2023-2024

## Step 1: Calibration (HS854231 semiconductors)
- HS9403 furniture: vessel=72.3%, air=1.9%, containerized=71.8%
- HS854231 semiconductors: vessel=0.9%, air=94.8%, containerized=0.9%
- Placebo check: Unexpectedly, semiconductor East-Coast vessel value ALSO shows a confirmed decline in this window -- this weakens confidence that a 9403 finding is Panama-Canal-specific rather than a broader East-Coast-wide effect unrelated to the canal (e.g. a different shared shock).

## Step 2: Validation (HS9403 furniture) -- country x coast-of-entry YoY confirmation
| Country | East status | East confirm month | West status | West confirm month | Pattern | Transport/port priority |
|---|---|---|---|---|---|---|
| China | confirmed | 2023-11-01 | not_confirmed | None | east_coast_specific_disruption | True |
| Vietnam | confirmed | 2024-01-01 | confirmed | 2023-10-01 | both_coasts_declined | True |
| Malaysia | confirmed | 2023-10-01 | confirmed | 2023-10-01 | both_coasts_declined | True |
| Taiwan | not_confirmed | None | confirmed | 2024-05-01 | west_coast_specific_decline_not_panama_related | False |
| Indonesia | confirmed | 2023-10-01 | confirmed | 2023-10-01 | both_coasts_declined | True |
| India | confirmed | 2023-11-01 | confirmed | 2023-10-01 | both_coasts_declined | True |
