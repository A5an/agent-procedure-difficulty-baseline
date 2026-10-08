"""Rubric prompts. Each takes the statement text (what an agent sees) and asks for one JSON object."""

STEP_TYPES = ["check_condition", "lookup_document", "ask_customer", "write_system",
              "irreversible_action", "branch", "refuse_or_handover"]

PROC = """You are a business-process analyst. Below is a customer request that a customer-service AI agent receives, together with the policy or domain description the agent works under (when the text contains a policy, use it; when it contains only a customer scenario, use the policy that such a service domain would normally apply, and say nothing about this).

Task: list the steps the agent must carry out to handle this request CORRECTLY according to the policy, in the order they would be done on the path the policy requires for this request. Label every step with exactly one type:

- check_condition: the agent tests a rule or eligibility condition from the policy against facts it already has or has just obtained (for example "membership tier is gold or silver", "the booking is within 24 hours"). Example: "Verify the account balance is at least the transfer amount".
- lookup_document: the agent reads information from a system, database record, policy text or knowledge base without changing anything. Example: "Retrieve the reservation details".
- ask_customer: the agent must obtain or confirm something from the customer (a missing parameter, an identity proof, an explicit confirmation). Example: "Ask the customer to confirm the cancellation".
- write_system: the agent changes system state in a way that can be corrected later. Example: "Update the shipping address on the order".
- irreversible_action: the agent performs an action that cannot be undone or has financial or legal effect. Example: "Issue the refund to the card", "Close the account".
- branch: the agent must choose between two or more different continuations that depend on an earlier result (the split itself, not the tests). Example: "If the fare is basic economy go to the no-change path, otherwise continue".
- refuse_or_handover: the agent declines the request or transfers the customer to a human. Example: "Tell the customer the cancellation is not allowed and offer a transfer".

For every step give "depth": the number of conditions that must hold for the step to be reached (0 for a step done in every case, 1 if it sits inside one condition, 2 if inside two nested conditions, and so on).
Also give:
- "refusal_conditions": the number of distinct conditions in the policy for this request that could make the correct outcome a refusal or handover.
- "needs_hidden_data": 1 if the correct decision depends on facts that are NOT stated in the customer message (for example the account record, a reservation, a database content), otherwise 0.
- "p_refusal": your estimate (0 to 1) of the probability that the correct outcome is a refusal (the request should not be carried out).

Return ONLY one JSON object, no other text, in this form:
{{"steps": [{{"text": "<short step>", "type": "<one of the seven types>", "depth": <int>}}, ...], "refusal_conditions": <int>, "needs_hidden_data": <0 or 1>, "p_refusal": <float>}}

=== REQUEST START ===
{text}
=== REQUEST END ==="""

LARA = """You are rating a business task for AI readiness with the five-dimension LARA rubric. The task is: "handle the request below correctly as a customer-service agent that can read and write business systems". Rate the task as a whole, using the policy or domain description in the text where present and otherwise the policy such a service domain would normally apply. Give each dimension an integer from 1 to 5.

D1 Cognitive complexity, anchored on Bloom's revised taxonomy: 1 = Remember (recall a fact), 2 = Understand (interpret or explain), 3 = Apply (use a rule or procedure in a given case), 4 = Analyze (compare, decompose, resolve conflicting information), 5 = Evaluate or Create (judge between alternatives, produce original judgement).
D2 Data dependency: 1 = a single structured source is enough; 2 = a few structured sources; 3 = several sources, some semi-structured; 4 = many sources with unstructured parts; 5 = real-time streaming plus unstructured external data.
D3 Interaction diversity: 1 = fully independent work; 2 = occasional contact with one party; 3 = regular contact with several parties; 4 = coordination across teams with differing interests; 5 = cross-organizational negotiation with political stakes.
D4 Compliance sensitivity: 1 = no compliance relevance; 2 = minor internal rules; 3 = clear policy or regulatory rules, errors need correction; 4 = strict rules, errors cause financial loss or audit findings; 5 = maximum, errors may cause legal liability.
D5 Innovation requirement: 1 = fully templated; 2 = template with small adaptations; 3 = standard solution adapted to the case; 4 = new combination of known solutions; 5 = breakthrough original solution required.

Return ONLY one JSON object, no other text:
{{"D1": <int>, "D2": <int>, "D3": <int>, "D4": <int>, "D5": <int>}}

=== TASK START ===
{text}
=== TASK END ==="""

RPA_CRITERIA = [
 ("standardization", "task", "degree of structure: every element unambiguous and the execution order the same in every instance"),
 ("maturity", "task", "the process flow is stable, specified and predictable over time, with few variants"),
 ("determinism", "task", "logical, rule-based execution steps that need no cognitive assessment or human judgement"),
 ("failure_rate", "task", "few failures, rework loops or non-recoverable terminations expected (a high score means a LOW failure rate)"),
 ("frequency", "time", "the activity is repeated often, in high transaction volumes"),
 ("duration", "time", "time to execute is short and predictable, so automation saves time"),
 ("urgency", "time", "the activity must be executed immediately or around the clock, where robots are better than people"),
 ("structuredness", "data", "the data is digital and at least semi-structured; unstructured, hardly accessible data scores low"),
 ("interfaces", "system", "the task can be solved through a system interface: few execution steps, little time spent in the interface"),
 ("stability", "system", "the involved systems and applications are stable and cause few exceptions"),
 ("number_of_systems", "system", "information is transferred between several systems, which adds value for a robot (a high score means automation potential from several systems)"),
 ("resources", "human", "many users perform the same task or many users contribute to one instance, so automation saves resources"),
 ("human_error_proneness", "human", "people are prone to errors on this activity (exceptions, time to resolve them), so a robot would help"),
]

RPA = """You are assessing how viable it is to automate a business task with robotic process automation, using the 13 criteria of the process characteristics evaluation framework of Wellmann et al. (2020). The task is: "handle the request below as a customer-service agent". Use the policy or domain description in the text where present, otherwise the policy such a service domain would normally apply. Several criteria are normally measured from event logs; here estimate them from the request and the kind of process it belongs to.

Score each criterion 0, 1 or 2 where 2 = the criterion strongly supports automation, 1 = partly, 0 = the criterion argues against automation. Criteria:
{criteria}

Return ONLY one JSON object, no other text, with the 13 criterion names as keys and the integer score as value:
{{{keys}}}

=== TASK START ===
{text}
=== TASK END ==="""

def rpa_prompt(text):
    crit = "\n".join(f"- {n} ({p} perspective): {d}" for n, p, d in RPA_CRITERIA)
    keys = ", ".join(f'"{n}": <0-2>' for n, _, _ in RPA_CRITERIA)
    return RPA.format(criteria=crit, keys=keys, text=text)

def proc_prompt(text):
    return PROC.format(text=text)

def lara_prompt(text):
    return LARA.format(text=text)
