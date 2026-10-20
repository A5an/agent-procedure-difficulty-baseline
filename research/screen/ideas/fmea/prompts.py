"""FMEA prompt. Category order is a parameter (used for the test-retest run)."""
CATS = ["skips_required_check", "acts_when_should_refuse", "refuses_when_should_act", "wrong_parameters_or_record",
        "incomplete_information_gathering", "wrong_calculation", "policy_misread", "premature_transfer_or_end"]
CAT_DEF = {
 "skips_required_check": "the agent performs the action without verifying a condition the policy requires",
 "acts_when_should_refuse": "the agent carries out a request that the policy forbids in this case",
 "refuses_when_should_act": "the agent declines or transfers a request that the policy allows in this case",
 "wrong_parameters_or_record": "the agent acts on the wrong record, id, item, amount or parameter value",
 "incomplete_information_gathering": "the agent does not collect or look up information it needs before deciding",
 "wrong_calculation": "the agent computes an amount, date, count or comparison incorrectly",
 "policy_misread": "the agent misunderstands or misapplies a rule of the policy",
 "premature_transfer_or_end": "the agent hands over to a human or ends the conversation before the case is resolved",
}
HEAD = """You are a quality engineer who runs a failure mode and effects analysis (FMEA, IEC 60812) on a business process. The process is: an AI customer-service agent (a capable large language model with tools) handles ONE customer case under a policy. The agent sees the policy and the customer message; it does not see the hidden database unless it calls a tool.

List up to 8 distinct ways this agent could fail on THIS case (fewer if fewer are realistic; do not pad). Each failure mode has:
- "category": exactly one of the categories below
- "occurrence": 1-10, how likely a capable LLM agent makes this error on this case
- "severity": 1-10, how bad the consequence is if it happens
- "detection": 1-10, how likely the error goes unnoticed (10 = unnoticed)
- "description": one short sentence

Categories:
{cats}

Scale anchors:
occurrence: 1 = almost never happens, 5 = happens in about half of the attempts, 10 = nearly every attempt fails this way.
severity: 1 = trivial, easily fixed, no customer impact; 5 = wrong outcome for the customer that needs a manual correction; 10 = irreversible financial or legal harm or a policy violation with liability.
detection: 1 = the error is obvious at once from the conversation or a system check; 5 = found only by reviewing the case afterwards; 10 = nothing would reveal the error, the outcome looks normal.

{policy}Case:
{text}

Return ONLY one JSON object: {{"modes": [{{"category": "...", "occurrence": 5, "severity": 5, "detection": 5, "description": "..."}}]}}"""

def prompt(text, order=None, policy=None):
    order = order or CATS
    cats = "\n".join(f"- {c}: {CAT_DEF[c]}" for c in order)
    pol = f"Domain policy the agent works under:\n{policy}\n\n" if policy else ""
    return HEAD.format(cats=cats, policy=pol, text=text)
