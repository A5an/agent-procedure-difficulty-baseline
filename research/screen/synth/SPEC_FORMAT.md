# How to write a procedure spec

You are writing one business procedure for a benchmark of customer-service AI agents. The agent will talk to a
simulated customer and use tools over a small database. A program reads your spec, computes the correct outcome of
every case from your rules, builds the tools and the database, and grades the agent by the final database state. So the
spec must be exact, and the policy text must say in plain language everything the rules check.

Make it feel like a real organization's procedure: realistic entities, field names, amounts, dates, people and
reasons for calling. Vary customers, names and situations across cases. Do not copy the structure of the example; fit
the structure to your domain and to the brief.

Output a single JSON object and nothing else (no markdown fences, no comments).

## Top-level keys

| key | what |
|---|---|
| `procedure_id` | copy from the brief |
| `domain` | copy from the brief |
| `organization` | a fictional organization name |
| `title` | short name of the procedure |
| `summary` | one sentence: what the procedure is for |
| `entities` | the database tables |
| `tools` | the agent's tools |
| `request` | the parameters of what the customer wants |
| `claims` | things the customer states that the policy checks against the database (identity checks); `{}` if none |
| `bindings` | the records the procedure looks at, in order |
| `outcomes` | the possible results, each with the exact write calls it makes |
| `rules` | ordered decision list; the first rule whose `when` is true decides the outcome |
| `default_outcome` | outcome when no rule fires |
| `policy_text` | the policy document the agent receives |
| `db_common` | records present in every case (catalogs, other customers); `{}` if none |
| `cases` | exactly the cases the brief lists |

## Types

`str`, `int`, `number` (money and other decimals), `bool`, `date` (string `YYYY-MM-DD`), `ref:<entity>` (the key of a
record of that entity), `enum:a|b|c` (one of the listed strings).

## entities

```
"entities": {"<entity>": {"key": "<key field>", "fields": {"<field>": "<type>", ...}, "description": "..."}}
```
Names are snake_case. The key field has type `str`. Every record must give every field (use `null` only if the value
is genuinely absent).

## tools

Four kinds. Every tool needs `name`, `kind`, `entity`, `description` (what it does, written for the agent).

- `get`: returns one record by key. Its only argument is the entity's key field.
  `{"name": "get_loan", "kind": "get", "entity": "loan", "description": "..."}`
- `search`: returns all records whose listed fields equal the given values (case-insensitive for text).
  `{"name": "find_loans_by_member", "kind": "search", "entity": "loan", "by": ["member_id"], "description": "..."}`
- `update`: changes fields of one record. Exactly one arg has type `ref:<its entity>` (the record to change). `set` maps
  fields to expressions over `args.<arg>`, `rec.<field>` (the record before the change) and `today`.
- `create`: adds a record (the program assigns its key). `fields` maps fields to expressions over `args.<arg>` and `today`.

Write tools (`update`, `create`) take `args`: `{"<arg>": {"type": "<type>", "description": "..."}}`. Write-tool args may
only be `ref:...`, `enum:...`, `int`, `number`, `bool` or `date`. No free text, because the grade compares database
states exactly. Every entity you bind needs a `get` tool. A `transfer_to_human_agents` tool is added automatically.

## request, claims, bindings

```
"request": {"loan_id": {"type": "ref:loan", "description": "the loan you want extended"},
            "days": {"type": "int", "description": "how many extra days you want"}},
"claims": {"birth_date": {"type": "date", "description": "your date of birth"}},
"bindings": [{"name": "loan", "entity": "loan", "key": "req.loan_id"},
             {"name": "mem", "entity": "member", "key": "loan.member_id"}]
```
Descriptions of request params and claims are read by the simulated customer, so write them from the customer's side
("your ...", "the ... you want").

The first binding is the main record of the request and is keyed by a request param. Later bindings are keyed by a
request param or by a `ref` field of an earlier binding. Every `ref` request param must key some binding.

## Expressions

Python-like: `and`, `or`, `not`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `+ - * / %`, `x if c else y`,
lists `[...]`, strings in single quotes, `True`, `False`, `None`. Names: `req.<param>`, `claim.<name>` (`None` if the
customer did not state it), `<binding>.<field>`, `today`. Functions: `days_between(a, b)` (days from date a to date b),
`add_days(d, n)`, `len`, `min`, `max`, `abs`, `round`, `lower`, and `db_count('<entity>', '<field>', value)` (number of
records of entity whose field equals value; needs a search tool on that entity by exactly that field). Ordering
comparisons with `None` are false. Text comparison with `==` ignores case.

## outcomes and rules

```
"outcomes": {"extend": {"kind": "perform", "description": "...",
                        "calls": [{"tool": "extend_loan", "args": {"loan_id": "req.loan_id", "days": "req.days"}}]},
             "review": {"kind": "perform", "description": "...",
                        "calls": [{"tool": "create_review_request", "args": {"loan_id": "loan.loan_id", "reason": "'hold_conflict'"}}]},
             "refuse": {"kind": "refuse", "description": "...", "calls": []}},
"rules": [{"id": "R1", "when": "claim.birth_date != mem.birth_date", "outcome": "refuse", "explain": "..."}, ...],
"default_outcome": "extend"
```
A `perform` outcome makes one or more write calls, in order, with args given as expressions; a `refuse` outcome makes
none. Every call gives exactly the tool's args. `explain` states the rule in one plain sentence.

## policy_text

The document the agent works from: how to identify the customer and the record, every rule (each `rules` entry must
be clearly stated, with its numbers and conditions), what to do in each outcome, and what the agent must not do. Write
it the way the organization would (headings and short paragraphs are fine), not as a copy of the expressions. It must
not contradict the rules, and it must not describe rules that are not in `rules`. The program adds the organization
name, the title and today's date on top.

## cases

```
{"case_id": "c1",
 "db": {"<entity>": [records]},
 "request_values": {"<param>": value},
 "customer_facts": [{"fact": "req.loan_id", "value": "L-88123", "disclose": "upfront"},
                    {"fact": "claim.birth_date", "value": "1990-04-12", "disclose": "if_asked"},
                    {"fact": "mem.email", "value": "dana.w@example.org", "disclose": "upfront"}],
 "extra_details": [{"text": "...", "kind": "helpful"}, {"text": "...", "kind": "distractor"}],
 "initial_request_values": {"days": 21},
 "persona": "...", "reason_for_call": "...", "task_instructions": "...",
 "expected_outcome": "extend"}
```
- `db`: this case's own records (the customer's records and anything else specific to the case). Together with
  `db_common` this is the whole database. Keys must be unique across both.
- `request_values`: what the customer really wants (after any change of mind). Ref values must exist in the database.
- `customer_facts`: everything the customer knows that matters. A fact is `req.<param>`, `claim.<name>` or
  `<binding>.<field>` (a fact about one of their records, e.g. the email on file, the title of the book, the balance
  they believe they have). `disclose`: `upfront` (they say it early on their own) or `if_asked` (only when the agent
  asks). Anything not listed is unknown to the customer. A `req.` fact must equal `request_values`. A fact about a
  record may differ from the database (the customer believes something wrong) only in `conflicting` cases.
- `extra_details`: other things the customer mentions, in plain words. `helpful`: true and relevant (it helps the agent
  find or check something). `distractor`: true but irrelevant to the decision (other products, old history, side
  stories, other requests they decide not to pursue).
- `initial_request_values`: only for behaviour `changes_mind`: what they ask for first (must differ from
  `request_values` in at least one param).
- `persona`, `reason_for_call`, `task_instructions`: the simulated customer's character, why they contact the
  organization, and how they behave in this conversation (follow the planned behaviour). Never put record ids in these
  three texts; ids reach the customer only through `customer_facts`. Names, places and stories are fine.
- `expected_outcome`: the outcome name you believe the rules give, or `cannot_complete` when the agent cannot identify
  the records or get the request params (the program checks this; it must match).

The program decides what the agent can reach: a binding is reachable if the customer states its key, or it is keyed
by a reachable binding's field, or a search tool finds exactly that one record from fields the customer knows
correctly (plus any other correct facts the customer gives about it, used to pick among several results). If some
record or request param cannot be reached, the correct outcome is `cannot_complete` and the correct calls are none.

## Planned information levels (the brief gives one per case)

- `upfront`: the customer knows the main record's id and every needed fact and says them on their own; no extra details.
- `on_request`: the customer knows the main record's id, but at least one needed fact is `if_asked`, so the agent must ask.
- `lookup`: the customer does not know the main record's id; the agent must find it with a search tool from other facts
  the customer knows (when several records match, the customer gives a fact that picks the right one).
- `insufficient`: the customer cannot give enough for the agent to identify the records (or a needed request param), so
  the correct outcome is `cannot_complete`. Make it natural (they lost the letter, called from a new number, etc.).
- `helpful_extra`: like `upfront`, plus at least two `helpful` extra details and at least one correct `upfront` fact about
  a field the rules check.
- `distracting`: the customer knows the main record's id; at least three `distractor` details; the case database holds at
  least two records of the main record's entity (e.g. several accounts), only one of which the request is about.
- `conflicting`: the customer knows the main record's id and states, upfront, at least one belief about a field the
  rules check that differs from the database (the agent must go by the database).

No level other than `conflicting` may contain a fact that contradicts the database.

## Planned behaviours

- `cooperative`: answers clearly and helpfully.
- `terse`: very short answers, gives only what is asked, never elaborates.
- `pushy`: impatient; if refused, argues and asks for an exception at least twice before accepting.
- `changes_mind`: first asks for `initial_request_values`, then switches before anything is done.

## Planned class

`perform` means the correct outcome makes write calls; `refuse` means it makes none (a `refuse` outcome or
`cannot_complete`). Pick facts and records so that your rules give the planned class and write that outcome in
`expected_outcome`.
