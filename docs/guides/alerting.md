# Alerting

ragdrift produces scores and `exceeded` flags. Turning those into pages is
your alerting backend's job — but a few patterns are worth flagging.

## Per-dimension thresholds

Different dimensions have different scales and different signal-to-noise
ratios. A reasonable starting set:

| Dimension   | Threshold | Notes |
|-------------|-----------|-------|
| Embedding   | 0.05      | MMD²+SW; bumps quickly when distributions diverge. |
| Data        | 0.10      | max(KS,PSI) per feature; PSI 0.10 is the standard moderate-shift line. |
| Response    | 0.20      | KS on lengths; sensitive to a 100-token shift on a 200-token base. |
| Confidence  | 0.20      | KS on score distribution; ECE-delta adds 0–0.5 when calibration breaks. |
| Query       | 0.10      | Symmetric KL on cluster assignments; 0.10 is "intent mix has noticeably shifted." |

These are starting points. Run the detectors against a quiet week of your own
production traffic, look at the score distribution, and pick thresholds at
roughly the 95th percentile.

## Don't page on every score change

A single window crossing the threshold is rarely actionable. Two patterns work:

- **K-of-N**: alert when at least K of the last N consecutive windows exceed.
- **Slow-burn**: alert when a 24h moving average exceeds.

Both filter the noise that comes from sample size, periodic traffic patterns,
and the inherent randomness of MMD's sample-pair construction.

## Composite alarms

If embedding drift fires *and* query drift fires, the upstream is probably the
issue (your users asked different questions). If embedding drift fires *and*
response drift fires but query drift does not, your retriever or generator
changed under you. CloudWatch and Datadog both let you express this as a
composite alarm.
