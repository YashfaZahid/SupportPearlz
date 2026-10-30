# SupportPearlz evaluation (20261029T202741Z)

## Aggregate
- cases: 18
- mean correctness: 0.94
- mean groundedness: 1.00
- mean citation accuracy: 0.78
- pass rate: 0.78
- refusal rate (unanswerable): 1.00
- hallucination proxy count: 0

## Results table
| id | category | verdict | corr | ground | cite | fault |
|----|----------|---------|------|--------|------|-------|
| E01 | easy | partial | 1.0 | 1.0 | 0.0 | citation |
| E02 | easy | pass | 1.0 | 1.0 | 1.0 | - |
| E03 | easy | pass | 1.0 | 1.0 | 1.0 | - |
| E04 | easy | pass | 1.0 | 1.0 | 1.0 | - |
| E05 | easy | partial | 1.0 | 1.0 | 0.0 | citation |
| M01 | multi_document | pass | 1.0 | 1.0 | 1.0 | - |
| M02 | multi_document | pass | 1.0 | 1.0 | 1.0 | - |
| M03 | multi_document | fail | 0.0 | 1.0 | 0.0 | generation |
| F01 | follow_up | pass | 1.0 | 1.0 | 1.0 | - |
| F02 | follow_up | pass | 1.0 | 1.0 | 1.0 | - |
| U01 | unanswerable | pass | 1.0 | 1.0 | 1.0 | - |
| U02 | unanswerable | pass | 1.0 | 1.0 | 1.0 | - |
| U03 | unanswerable | pass | 1.0 | 1.0 | 1.0 | - |
| A01 | ambiguous | pass | 1.0 | 1.0 | 1.0 | - |
| A02 | ambiguous | pass | 1.0 | 1.0 | 1.0 | - |
| V01 | adversarial | pass | 1.0 | 1.0 | 1.0 | - |
| V02 | adversarial | partial | 1.0 | 1.0 | 0.0 | citation |
| V03 | adversarial | pass | 1.0 | 1.0 | 1.0 | - |

## Failure analysis
- **E01** (partial): stage=`citation` — Return 24 months from invoice date with warranty citation.
- **E05** (partial): stage=`citation` — Map to E-03 sediment filter; replace and RESET 5 seconds.
- **M03** (fail): stage=`generation` — Within period but unauthorised repair is an exclusion; point to paid service.
- **V02** (partial): stage=`citation` — Contradict false premise; state documented 24 months.
