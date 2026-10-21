import json
from lib import DEMANDS, compact_rubrics
from checklists import CHECK
R = compact_rubrics()
RUB = "\n\n".join(R[d] for d in DEMANDS)
INTRO = ("You rate the demands that an AI agent faces when it handles a business-procedure task. The TASK INSTANCE below is what the agent "
         "receives (a policy for an action, a customer message or scenario, and available tools). Rate how demanding the task is on 18 scales, "
         "using the level descriptions of each scale (levels 0 to 5).\n\n")
def prompt_A(text):
    return (INTRO + "SCALES AND LEVELS:\n" + RUB + f"\n\nTASK INSTANCE:\n{text}\n\n"
            "For each of the 18 scales give your probability for each level 0,1,2,3,4,5 (six numbers that sum to 1). "
            "Spread probability when you are unsure which level fits. Answer with one JSON object only, no other text, for example "
            '{"AS":[0.1,0.6,0.3,0,0,0],"AT":[...],...} with all 18 keys: ' + ",".join(DEMANDS) + ".")
def prompt_INT(text):
    return (INTRO + "SCALES AND LEVELS:\n" + RUB + f"\n\nTASK INSTANCE:\n{text}\n\n"
            "For each of the 18 scales choose the single integer level 0 to 5 that fits best. Answer with one JSON object only, no other text, e.g. "
            '{"AS":1,"AT":2,...} with all 18 keys: ' + ",".join(DEMANDS) + ".")
def prompt_B(text):
    qs = []
    for d in DEMANDS:
        for i, q in enumerate(CHECK[d]): qs.append(f"{d}{i+1}. {q}")
    return ("An AI agent must handle the TASK INSTANCE below (a policy for an action, a customer message or scenario, and available tools). "
            "Answer each yes/no question about the demands of the task. 'The solver' means the agent. Answer true only if the task really demands it.\n\n"
            f"TASK INSTANCE:\n{text}\n\nQUESTIONS:\n" + "\n".join(qs) +
            "\n\nAnswer with one JSON object only, no other text, mapping each question id to true or false, e.g. {\"AS1\":true,\"AS2\":false,...} with all 90 ids.")
def prompt_C(text, domain, anchors):
    """anchors: list of (text, levels dict)"""
    a = ""
    for i, (t, lv) in enumerate(anchors):
        a += f"ANCHOR TASK {i+1}:\n{t}\nReference levels: {json.dumps({d: lv[d] for d in DEMANDS})}\n\n"
    return (INTRO + "SCALES AND LEVELS:\n" + RUB + f"\n\nThe following {len(anchors)} anchor tasks come from the same domain ({domain}) and are already rated. "
            "Use them as fixed reference points: rate the new task relative to them on every scale (higher than an anchor if it demands more on that scale, "
            "lower if less, equal if similar). You may use half levels such as 2.5.\n\n" + a +
            f"NEW TASK INSTANCE:\n{text}\n\nAnswer with one JSON object only, no other text, with the level 0 to 5 of the new task for all 18 keys: " + ",".join(DEMANDS) + ".")
