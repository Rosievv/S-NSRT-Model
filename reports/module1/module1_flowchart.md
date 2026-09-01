# Module 1 Flowchart

```mermaid
flowchart TD
    A[Start Module 1 Script] --> B[Load Source Data\n data/raw/us_census_*.parquet]
    B --> C[Concatenate and Deduplicate Rows]
    C --> D[Convert date to datetime]
    D --> E[Filter Aggregate Labels\n e.g. APEC, ASEAN, Europe, Total]

    E --> F[Build SupplyChainNetwork]
    F --> G[Build Directed Graph\n country -> USA\n weight = value_usd]
    G --> H[Compute Centrality Metrics\n market share, PageRank, betweenness, eigenvector]
    H --> I[Identify Critical Nodes\n threshold >= 5% share]

    E --> J[Initialize StressTestRunner\n substitution_elasticity = 0.3]
    J --> K[Run Scenario Library]
    K --> K1[top_1_supplier_failure]
    K --> K2[top_3_supplier_failure]
    K --> K3[east_asia_moderate]
    K --> K4[east_asia_severe]
    K --> K5[china_decoupling]
    K --> K6[taiwan_crisis]

    K1 --> L[Per-HS Shock Calculation]
    K2 --> L
    K3 --> L
    K4 --> L
    K5 --> L
    K6 --> L

    L --> M[Compute Results\n direct loss, substitution, net loss, supply gap %]
    M --> N[Convert to Summary DataFrame]

    J --> O[Backtest Historical Events\n COVID 2020, Japan 2011, Thailand 2011]
    O --> P[Compare Simulated Gap vs Observed Change]

    N --> Q[Save Outputs]
    P --> Q
    I --> Q

    Q --> Q1[module1_stress_results.json]
    Q --> Q2[module1_stress_summary.csv]
    Q --> Q3[module1_backtest.csv]
    Q --> Q4[module1_centrality.csv]
    Q --> Q5[Charts in reports/module1/figures]

    Q5 --> R[End]
    Q4 --> R
    Q3 --> R
    Q2 --> R
    Q1 --> R
```

## Notes

- Source data is US Census trade parquet files loaded from data/raw.
- Core modeling is HS-level shock propagation with substitution limits.
- Scenario conclusions are based mainly on supply_gap_pct and substitution_absorbed_pct.
