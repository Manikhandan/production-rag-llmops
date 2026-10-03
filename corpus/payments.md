# Payments retry policy

Failed card captures may be retried twice with exponential backoff.
Do not retry after a hard decline from the issuer.
Operators page the payments on-call after the second failure.
Capture logs must include the decline code and the merchant identifier.
