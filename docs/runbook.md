# Incident: retrieval returns empty citations

Check corpus path, chunk count on `/ready`, and whether the question terms exist in the markdown files.

# Incident: hallucination_flag true

The extractive provider should not trip this. If a remote provider is wired later, treat the flag as a eval signal, not a page, until it is calibrated.
