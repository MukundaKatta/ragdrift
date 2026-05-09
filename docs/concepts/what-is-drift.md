# What is drift?

Drift is the gap between the data your system was designed for and the data it
sees today. RAG drift specifically takes five overlapping forms:

1. **The user is asking different questions** — the input distribution shifted.
2. **The retriever is finding different documents** — the corpus shifted, or the
   embedding model was updated, or both.
3. **The model is producing different answers** — length, tone, semantic content.
4. **The model is more or less confident than before**, regardless of whether
   the answers are still right.
5. **The shape of the workload has changed** — where last week was 70% billing
   questions, this week is 70% returns.

Each of these can degrade end-user experience without any explicit error. They
do not surface as 5xx responses, they do not fail tests, and they do not fire
your latency alerts. They are silent.

The textbook approach is to catch the problem at evaluation time using a held-out
set. That works for the next training cycle. It does not help you on Tuesday at
3 PM when your retriever quietly started returning slightly worse results because
your corpus drifted.

`ragdrift` watches all five dimensions in production, in milliseconds, with a
Rust core so it does not become its own latency tax.

## What ragdrift does *not* do

- It does not retrain or re-embed your model. That is your problem.
- It does not tell you *why* a dimension drifted. The score and the dimension
  name are signals; root cause is for you and your dashboards.
- It does not predict drift before it happens. Forecasting is a different problem.
