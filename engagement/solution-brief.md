# Harbourline Support Email Intake — Solution Brief

## Page 1: Week 1, "Email Indigestion"

### The problem: raw emails are not tickets
Harbourline support receives unstructured emails: coverage questions, billing disputes, claim status requests and general messages. Before an agent can act on one, they have to work out what it is and whether it can be answered yet:
- **Identifiers are often missing.** A coverage question with no `policy_number`, or a claim question with no `claim_id`, can't be handled until the agent asks for it.
- **Wording varies across languages.** One email may mix two languages for the same request.
- **Customers repeat conflicting statements** they heard elsewhere ("a friend said I need a referral"), and the reply must neither confirm nor deny them without the policy in hand.

Today the agent does all of this triage by hand, email by email.

### Primary user
**A Harbourline support agent.** They review incoming emails, ask customers for missing identifiers, and send final replies. The pilot prepares the work, and the agent stays accountable for every message that reaches a customer.

### What Week 1 delivers
One email in, one validated `Intake` object out:

| Field | Meaning |
|---|---|
| `category` | `coverage`, `billing`, `claim` or `other` |
| `missing_fields` | required identifiers absent from the email, e.g. `["policy_number"]` |
| `next_action` | `ask` (request the missing identifiers) or `draft` (proposed reply ready for review) |
| `customer_message` | the clarifying question or the proposed draft reply |
| `confidence` | 0.0–1.0 |

Every model response is checked in code against this schema. An invalid response is sent back to the model with the exact error. This stops after at most 3 model calls per email, and every attempt, error and stop reason is logged.

### Autonomy boundary: **Draft**
The system **classifies and proposes**. The agent **decides and sends**.
- The system produces triage data and a *proposed* customer message.
- A support agent reads, edits, approves or discards every message.
- The system never states coverage, referral or eligibility rules, amounts owed, or claim outcomes. Its drafts say an agent will confirm.

### Explicit exclusions (Week 1)
1. **No sending emails to customers.** Nothing leaves Harbourline without an agent.
2. **No paying or adjudicating claims.** There is no claim decision, approval, denial or payment of any kind.
3. **No automated ticket/CRM writes.** Output is local JSON for review, and integrations come in Week 3.
4. **No policy handbook retrieval / RAG.** The model has no access to policy rules, by design, until Week 2.
5. **No web UI.**

### Week 1 acceptance (sample emails)
| # | Email | Expected |
|---|---|---|
| 1 | Priya: dermatologist / specialist question, no policy number | `coverage`, `missing_fields` ⊇ `["policy_number"]`, `ask`, no invented coverage or referral rules |
| 2 | Billing: duplicate charge, with a policy number and dollar amount | `billing`, `draft` |
| 3 | Claim status, with a claim ID | `claim` |
| 4 | Thank-you note | `other` |
| 5 | Bilingual (Spanish/English) specialist question | `coverage` |
