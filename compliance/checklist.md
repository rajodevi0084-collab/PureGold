# Compliance Checklist for Indian Capital Markets Operations

This checklist summarises mandatory controls for strategies executed through Indian exchanges while interfacing with KiteConnect. Each item should be reviewed before deployment and revalidated quarterly.

## SEBI Regulatory Guidelines
- Maintain an active registration under SEBI (Investment Adviser/Portfolio Manager) with up-to-date Form C filings.
- Publish a detailed risk disclosure document to clients and secure signed acknowledgements prior to execution.
- Enforce client-level suitability assessments and record-keeping as per SEBI (Investment Advisers) Regulations, 2013.
- Implement surveillance for front-running, price manipulation, and circular trading. Escalate suspicious alerts to the compliance officer immediately.
- Retain trade blotters, contract notes, and investor communications for a minimum of eight years.

## Data Usage and Licensing
- Use only market data licensed for the intended purpose (live trading, delayed display, redistribution). Document the licence terms alongside the data source.
- Restrict raw exchange feeds to authorised personnel. Mask or aggregate data before sharing with research contractors.
- Track data retention periods mandated by the vendor and purge expired datasets from primary and backup storage.
- Ensure survivorship-bias-free symbol masters and index constituent files are refreshed in accordance with exchange corporate action bulletins.

## Audit Logging and Record Keeping
- Centralise API order logs, acknowledgements, and error responses with immutable timestamps.
- Capture configuration changes (risk limits, credentials, routing rules) in an append-only audit trail.
- Version-control strategy code, parameter files, and deployment manifests; retain tagged releases for regulatory inspection.
- Provide daily reconciliation reports (trades vs. broker confirms) signed off by both the dealing desk and compliance officer.

## Operational Controls
- Validate that disaster recovery procedures satisfy SEBI Business Continuity Planning mandates, including annual failover testing.
- Maintain client data privacy in line with the Information Technology (Reasonable Security Practices and Procedures and Sensitive Personal Data or Information) Rules, 2011.
- Conduct semi-annual penetration tests and document remediation for high or critical findings.
- Review third-party vendor SLAs to ensure they include breach notification and regulatory cooperation clauses.

Use this checklist as part of pre-trade compliance reviews and archival packages prepared for statutory audits.
