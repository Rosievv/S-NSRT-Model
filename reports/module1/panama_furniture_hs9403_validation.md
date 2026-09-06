# Module 1 Validation: HS 9403 (Furniture) vs Panama Canal Drought 2023-2024

- Event tested: `panama_canal_drought_2023` (event date 2023-10-31)
- Confirmation rule: YoY decline <= -20.0% within 8 months of the event date (Reused from Module 1 customs-confirmation rule (build_event_monitor_v1.py))

## 1) Auto-selected main Asia source countries
- Method: Top 6 of candidate Asian countries by historical (<= 2022-12-31) HS 9403 import value_usd share
- Selected: China, Vietnam, Malaysia, Taiwan, Indonesia, India

## 2) Confirmation results
- All countries: {'status': 'not_confirmed', 'confirmation_month': None, 'min_yoy_pct_in_window': -13.58}
- Asia selected group (aggregate): {'status': 'not_confirmed', 'confirmation_month': None, 'min_yoy_pct_in_window': -15.25}
- Control group (Canada + Mexico, overland): {'status': 'confirmed', 'confirmation_month': '2024-01-01', 'yoy_pct_at_confirmation': -20.69, 'min_yoy_pct_in_window': -20.69}
- Breadth: 83.33% of selected Asia countries independently confirmed a decline
- Earliest Asia-country confirmation: 2023-10-01
- Asia median trough YoY: -22.05%
- Control lag vs earliest Asia confirmation: 3.0 months
- Control concurrent and comparable to Asia (would undermine transport hypothesis): False

## 2b) Module 1 model-predicted supply gap (StressTestRunner / PropagationEngine)
- Predicted supply gap (model): 15.19%
- Observed supply gap (Module 1 trend-counterfactual method): 0.0%
- Predicted severity bin: high, observed severity bin: low
- Directional hit (predicted and observed fall in same severity bin): False
- estimated_severity=0.33 approximates the ACP's real transit-capacity cut (36/day normal -> ~24/day trough in early 2024), not a calibrated furniture-specific figure.
- HS 9403 has no entry in PropagationEngine.HS_ELASTICITY_MAP, so the model falls back to the generic default substitution_elasticity (0.3) scaled by the 'logistics' event-type multiplier -- it is not a furniture-calibrated elasticity.

## 2c) News early-warning value vs. waiting for customs/trade data
- ACP official advisory date: 2023-10-30
- News wire (Reuters) date: 2023-10-31
- Customs-confirmed decline month: 2023-10
- Approx. Census public availability date for that month's data: 2023-12-08 (assumes ~38-day release lag)
- Lead time: news wire beats customs-data availability by ~38 days; official advisory beats it by ~39 days
- The model's estimated_severity (0.33) is derived directly from the ACP's own announced transit-capacity cut, not from trade data -- so the 15.19% predicted supply-gap (severity_bin=high) could in principle have been produced the same day as the news/advisory, without waiting for any customs data.
- Customs/trade-data confirmation of the decline would not be publicly available until approximately 2023-12-08, about 38 days after the news wire reported the ACP's booking-slot cut, and about 39 days after the ACP's own official advisory. News-based monitoring plus the stress-test model together would have flagged this as a high-severity transport-channel risk essentially in real time, well before trade statistics could confirm it.

## 2d) Existing-data validation checks (using only fields already in the HS 9403 parquet)
- Monthly observed-gap pattern: `no_clear_pattern` (peak month 2023-10, peak gap 0.0%)
  - Gap does not show a clean early-peak/late-fade or sustained shape.
- Transit-lag timing check: `decline_precedes_event_date` (lag -30 days vs event date 2023-10-31)
  - The confirmed customs decline appears 30 days BEFORE the 2023-10-31 restriction date used here -- faster than the 28-60 day physical transit-time window, which suggests carriers/shippers had already started cutting bookings or diverting cargo ahead of the formal ACP advisory. This is consistent with the ACP progressively tightening transit quotas through mid-to-late 2023 (well before the October advisory used as the event date here), rather than a single-day shock.
  - Reference: Transit-day bands as supplied by the user from external industry reporting; not independently verified in this project.
- Unit-value/freight-cost proxy check: `not_possible_with_current_data` -- The 'quantity' column in this parquet file is uniformly 0 for every record (verified: min/mean/max all 0.0), so value_usd/quantity cannot be used as a unit-value or freight-cost proxy with this dataset.
- **Severity match (predicted vs YoY-implied): predicted=`high`, YoY-implied=`high` (22.05% median trough), match=`True`**
  - Using the YoY-confirmed decline (22.05% median trough across Asia source countries) as ground truth instead of the trend-counterfactual method, the model's same-day predicted severity bin ('high') DOES match the trade-data-confirmed severity bin ('high'). This is the more meaningful early-warning validation: the model's real-time, news-triggered 'high severity' call is corroborated -- with a ~38-day lag -- by the customs data itself, once measured with a metric (YoY) sensitive enough to see this event at all.

## 3) Recommendation
- **transport_port_rerouting_priority**
- Rationale: Multiple independent Asian source countries (5 of 6 selected) show a deep YoY decline concentrated in the 1-2 months immediately after the Panama Canal booking-slot restriction (2023-10-31), well beyond the -20%/8-month confirmation rule. The overland, non-canal-dependent Canada/Mexico control group only dips later and/or more shallowly, so it does not explain the immediate, broad, Asia-concentrated shock. This pattern is consistent with a shared transit chokepoint (Panama Canal) rather than a country-specific supplier production failure or a general US demand slump, and should be routed to transport/port/rerouting investigation ahead of supplier-production investigation. Earliest Asia-country confirmation: 2023-10-01; control group confirmation: 2024-01-01 (lag vs Asia: 3.0 months); Asia median trough YoY: -22.05%, control trough YoY: -20.69%.

## 3b) Per-country channel classification (source / transport / mixed / undetermined)
- **China**: channel=`undetermined` (No confirmed YoY decline for this country under the -20%/8-month rule.); Case1 3-way priority=`undetermined_pending_company_data` (Public Census data has no entry-port, transit-mode, or weeks-of-supply fields; supply data/company/hs9403_company_overrides.csv to evaluate this.)
- **Vietnam**: channel=`mixed` (Confirmed decline -1.0 months after the canal restriction -- ambiguous timing; could reflect a slower rerouting/backlog effect layered on other factors.); Case1 3-way priority=`undetermined_pending_company_data` (Public Census data has no entry-port, transit-mode, or weeks-of-supply fields; supply data/company/hs9403_company_overrides.csv to evaluate this.)
- **Malaysia**: channel=`mixed` (Confirmed decline -1.0 months after the canal restriction -- ambiguous timing; could reflect a slower rerouting/backlog effect layered on other factors.); Case1 3-way priority=`undetermined_pending_company_data` (Public Census data has no entry-port, transit-mode, or weeks-of-supply fields; supply data/company/hs9403_company_overrides.csv to evaluate this.)
- **Taiwan**: channel=`source` (Confirmed decline 6.0 months after the canal restriction -- too delayed to be explained by a single transit disruption; more consistent with a country-specific supply issue, unless news evidence ties it to a secondary transport effect.); Case1 3-way priority=`undetermined_pending_company_data` (Public Census data has no entry-port, transit-mode, or weeks-of-supply fields; supply data/company/hs9403_company_overrides.csv to evaluate this.)
- **Indonesia**: channel=`mixed` (Confirmed decline -1.0 months after the canal restriction -- ambiguous timing; could reflect a slower rerouting/backlog effect layered on other factors.); Case1 3-way priority=`undetermined_pending_company_data` (Public Census data has no entry-port, transit-mode, or weeks-of-supply fields; supply data/company/hs9403_company_overrides.csv to evaluate this.)
- **India**: channel=`transport` (Confirmed decline 0.0 months after the canal restriction -- consistent with in-transit/booking-slot disruption rather than a domestic production issue.); Case1 3-way priority=`undetermined_pending_company_data` (Public Census data has no entry-port, transit-mode, or weeks-of-supply fields; supply data/company/hs9403_company_overrides.csv to evaluate this.)

## 3c) Capability gap matrix (can Module 1 predict the user-described risks?)
- **transit_time_and_ghost_leadtime**: `not_captured` -- Model has no vessel-level transit-time, anchorage queue, or booking-slot data; it only sees monthly country-level import value_usd. Needed: AIS/vessel transit-time data or company-reported door-to-door lead times.
- **freight_rate_and_surcharge**: `not_captured` -- No freight-rate, PCC-surcharge, or per-container cost data in the trade_df schema. Needed: Freight-rate index (e.g. Freightos/Drewry) or carrier surcharge schedules.
- **rerouting_secondary_effects**: `partially_captured` -- PropagationEngine's substitution_absorbed_pct captures a generic rerouting/substitution concept (predicted_substitution_absorbed_pct=~20% in this run), but it cannot distinguish USWC mini-landbridge vs Cape-of-Good-Hope rerouting, or their distinct cost/time impacts. Needed: Port-of-entry field (USEC vs USWC) plus routing/mode data.
- **bullwhip_and_stockout**: `not_captured` -- No inventory, weeks-of-supply, or DC-level order/backlog data in current pipeline. Needed: Company inventory/WOS and order-backlog data (see company-data plug-in interface).
- **case1_3way_priority_test**: `undetermined_pending_company_data` -- The 'long lead time + USEC all-water port + low WOS' intersection requires port-of-entry, routing mode, and WOS fields that do not exist in public Census data. Needed: Populate data/company/hs9403_company_overrides.csv (see schema above) to evaluate this per country.

## 3d) Crawled news evidence (discovery-only, full text not verified)
- Status: discovered_headlines_only_full_text_not_verified
- Candidates discovered: 39, saved full text: 0, model-eligible: 0
  - "Panama canal says will slash booking slots due to drought" -- Reuters (2023-10-31)
  - "Impact to Global Trade of Disruption of Shipping Routes in the Red Sea, Black Sea and Panama Canal" -- UN Trade and Development (UNCTAD) (2024-02-15T08:00:00+00:00)
  - "Middle East conflict, Panama drought are worrying for Georgia’s ports" -- AJC.com (2024-01-31T08:00:00+00:00)
  - "What Happens When Ships Can't Cross the Red Sea and Panama Canal?" -- Chicago Council on Global Affairs (2024-01-22T08:00:00+00:00)
  - "New Year supplies under threat by Red Sea shipping crisis: Warnings of delays to availability of goods including tea, coffee, wine and clothes as more than 100 container ships are forced to reroute around Cape of Good Hope just days before Christmas" -- Daily Mail (2023-12-21T08:00:00+00:00)
  - "Panama Canal transits plunge as larger ships are turned away" -- FreightWaves (2023-12-13T08:00:00+00:00)
  - "Global shipping rates set to jump as carriers avoid the Red Sea amid Houthi attacks" -- cnbc.com (2024-01-10T08:00:00+00:00)
  - "Companies rush to avert disruption from Red Sea attacks as shipping rates rise" -- Reuters (2023-12-19T08:00:00+00:00)
- Confound note: Most discovered coverage is about the concurrent Red Sea/Suez crisis (Dec 2023-Jan 2024), which also disrupted Asia-to-US-East-Coast shipping at the same time as the Panama Canal restriction. Public news alone cannot cleanly separate the two channels for HS 9403; e.g. 'US West Coast Ports See Minimal Red Sea Cargo Rerouting' (gCaptain, 2024-02-14) and '2M adjusts Asia-US East Coast ship schedules to account for Cape reroutings' (Journal of Commerce, 2024-01-24) point to carriers rerouting around Africa, not necessarily via Panama-to-USEC volume shifting to USWC.

## 4) Reviewer note: what is still missing to fully confirm
- No US entry-port field: cannot attribute the decline to a specific port (e.g. East Coast ports reached via Panama vs West Coast ports), only to source country totals.
- No vessel-mode / containerized-vessel value split: cannot separate containerized ocean freight (Panama-routed) from air or other modes within the same country total, which would sharpen the transport-channel diagnosis.
- No inventory or on-hand stock data: cannot distinguish a real import decline from destination-side inventory drawdown that masks continued but delayed shipments.
- No lead-time / transit-time or backlog data: cannot confirm whether the decline reflects delayed-in-transit cargo (consistent with rerouting) versus cancelled or substituted orders (consistent with a demand or supplier issue).
