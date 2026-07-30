# Requirements

## Purpose

A citation-grounded RAG system for searching medical guidelines. It
helps healthcare professionals locate relevant passages from medical
guideline documents.

This system is a technical demonstration. It must not provide medical
diagnoses, individual treatment decisions, or patient-specific
recommendations.

## Functional Requirements

- Search medical guideline documents using natural language.
- Return answers grounded only in retrieved passages.
- Show citations with document title, edition, chapter, section, and
  page.
- Return an insufficient-evidence response when reliable evidence is
  absent.

## Non-Functional Requirements

- Separate retrieval quality from generation quality.
- Provide a reproducible environment using Docker.
- Deploy the application to AWS.
- Maintain production-oriented code quality.

## Out of Scope (for now)

- Patient-specific diagnosis or treatment recommendations.
- Storage of real patient information.

See `CLAUDE.md` for the full set of project rules.
