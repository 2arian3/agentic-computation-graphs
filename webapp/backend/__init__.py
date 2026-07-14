"""ACG experiment dashboard — backend package.

A thin FastAPI layer over the existing `acg` instrument. It imports `acg` read-only
(the only writes are the document manager editing data/corpus.json, which is its job)
and adds nothing to the experimental logic. See docs/09-webapp-architecture.md.
"""
