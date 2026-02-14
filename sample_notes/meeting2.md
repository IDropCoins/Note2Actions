# Engineering Standup - Feb 13

Attendees: Krish, Neha, Ravi

## Updates
- CI pipeline now runs unit tests in parallel.
- Two flaky tests were quarantined pending fixes.
- Background worker memory usage improved after patch.

## Risks
- Third-party API rate limits may affect daily sync jobs.
- Mobile QA is behind by one day.

## Next Steps
- Neha: unflake payment webhook tests.
- Ravi: add retry/backoff for sync worker.
- Krish: coordinate with QA on release checklist.
