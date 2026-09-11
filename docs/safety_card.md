# Safety card

## Defense-only

ReturnGuard AI detects, explains, and helps **prevent merchant loss**. It does not teach or automate attacks.

## Prohibited

- Stolen card / OTP / credential generation or testing
- Exploiting refund, payment, or identity systems
- Auto-cancel, blacklist, or silent denial of service to a customer
- Protected attributes (race, religion, gender, caste, exact age) as features
- Claiming a score is proof of criminal fraud

## Fail closed

If the model artifact cannot load, `POST /api/v1/score` returns HTTP 503 and instructs **manual review**. The API will not invent a probability.

## Privacy

Demo identifiers are pseudonymous (`ORD-#####`). No names, emails, phones, exact addresses, or PAN/card data are stored as features.

## RBAC (demo)

The UI is a single demo operator. A production deployment must add authentication, roles (admin / analyst / operator / read-only), TLS, and secret management.
