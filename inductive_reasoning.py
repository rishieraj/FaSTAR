# ------------------ inductive_learning.py -------------------------------------
"""
Online sub‑routine induction & refinement (FaSTA* §4.2  +  App. D/O).

Usage
-----
• import append_trace, refine_rules_if_needed anywhere in the pipeline
  (three one‑liners shown below).
• Nothing else to do – the learner runs invisibly in the background.
"""
from __future__ import annotations
import json, time, uuid, os
from pathlib import Path
from openai import OpenAI

# -----------------------------------------------------------------------------#
TRACE_BUF        = Path("trace_buffer.jsonl")      # rolling buffer of raw traces
CUSTOM_SRT       = Path("srt_custom.json")         # refined rules live here
REFINE_FREQ_K    = 20                              # §K.1, line 2 :contentReference[oaicite:0]{index=0}
MODEL_NAME       = "gpt-4o-mini"                   # cheap/fast is enough
# -----------------------------------------------------------------------------#

# ---------- helpers -----------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def _write_json(path: Path, obj): path.write_text(json.dumps(obj, indent=2))

def _load_rules() -> dict:  return json.loads(CUSTOM_SRT.read_text()) if CUSTOM_SRT.exists() else {}

# ---------- public API --------------------------------------------------------

def append_trace(trace: dict) -> None:
    """Call **once per completed task**, e.g. right after `run_subsequence`."""
    with TRACE_BUF.open("a") as f:
        f.write(json.dumps(trace)+"\n")

def refine_rules_if_needed(llm_key: str) -> bool:
    """
    Triggered after every task; runs only when ≥ K traces are buffered.
    Returns True if CUSTOM_SRT has been updated.
    """
    traces = _load_jsonl(TRACE_BUF)
    if len(traces) < REFINE_FREQ_K:
        return False                      # nothing to do yet

    client = OpenAI(api_key=llm_key)
    rules  = _load_rules()

    # --- the exact system/user prompt from App. O (truncated here for brevity) --
    system_prompt = f"""Analyze the provided experimental run data for a specific task (e.g., Object Recol-
oration) to infer initial, potentially qualitative, activation rules (preference conditions) for
each distinct execution path employed.
<All Logged Data (The Traces)>
So we run different models and tools for different or same image editing tasks and store the
observations including what path was finally used and what were the conditions of objects, etc.
and this data is provided to you. Now we wish to infer some subroutines or commonly used
paths and their activation rules under which they are commonly activated. Can you find some
commonly used subroutines or paths and infer some rules for these paths using the status
of these cases and other factors and give the rules for both paths and they need not be too
specific but a bit vague is fine like if you observe that some particular path always fails in case
object size is less then you can give the rule that this path should be used when object is not
too small and not give any specific values so activation rule will include like object_size
= not too small, etc like this based on all factors like object size, color transitions, etc
and also it is possible that for some path it failed bcz of some specific condition like its not
necessary all conditions led to failure so you need to check which is the condition which
always leads to failures or which always leads to success and that will constitute a rule if
some condition leads to both failures and success with same value then it means that this is
not the contributing factorand there’s something else that’s causing the failure or success and
keep in mind that output rules should be of activation format like in what cases this should be
used and not negative ones so if there is some path which always fails when object size is
big then your activation rule will have object_size = small and not some deactivation
rules which has object_size = big. You should also include some explanatory examples
in the rule which can help some person or LLM understand them better when referring to
these rules. eg. if there is a rule where you want to say that this path will only succeed
when the difference between size of objects is not too big then you can have a rule like
: “size_difference(original, target objects) = Not too big (eg. hen to
car, etc)” where you include some example. You should focus on activation rules which
are like in what case this particular path will always succeed and some activation rules should
also include a kind of deactivation rule with a not like in case you observe that some path
always fails when there is some condition x where x can be like object is too small or color
difference is huge then you should infer an activation rule that is negate of this like the rule
can be object is “not” too small or color difference is “not” huge so that these activation rules
can act as a kind of deactivation rules as well and prevent the path from getting activated in
cases where we know for sure it’ll fail.
An Example:
Experimental Data:
Subtask: s1, Object Size: 0.7px, Original Object Color: Yellow, Target
Object Color: Black
Path used: P1
Status: Fail
Subtask: s1, Object Size: 0.2px, Original Object Color: Yellow, Target
Object Color: Green
Path used: P1
Status: Success
Subtask: s1, Object Size: 5px, Original Object Color: White, Target
Object Color: Black
Path used: P1
Status: Fail
Subtask: s1, Object Size: 5px, Original Object Color: White, Target
Object Color: Yellow
Path used: P1
Status: Success
Subtask: s1, Object Size: 0.7px, Original Object Color: Black, Target
Object Color: White
Path used: P1
Status: Fail
Subtask: s1, Object Size: 3px, Original Object Color: Black, Target
Object Color: Yellow
Path used: P2
Status: Success
Subtask: s1, Object Size: 0.9px, Original Object Color: Black, Target
Object Color: Blue
Path used: P2
Status: Fail
Subtask: s1, Object Size: 0.2px, Original Object Color: White, Target
Object Color: Yellow
Path used: P2
Status: Fail
Subtask: s1, Object Size: 0.6px, Original Object Color: White, Target
Object Color: Blue
Path used: P3
Status: Success
So we see that paths P1 and P2 are very commonly used so these will be our subroutines or
commonly used paths and now our goal is to infer some rules under which these subroutines
or paths are commonly activated. So by observing the data you see that in P2 it always fails
when object size is small while the color transitions doesn’t matter so for P2 you can infer an
activation rule which is “object_size: not too small” and while for P1 you observe
that the object size doesn’t really matter bcz it is able to succeed in both small and big object
sizes and also fail in both cases but you observe that when the color transition is huge like
white to black or black to white it always fails while when color transition is not extreme like
white to yellow or yellow to green it is able to succeed even under same size conditions so
you can infer a rule that it depends on color transition and give a rule with example for better
understanding like: “color_transition: not too extreme (eg. not white <->
black, etc.)”
The real experimental data will include much more info and it is your job to infer what data
is useful and find patterns in it and give corresponding rules. Also you should not mix the
observations from different paths or subtasks and treat all paths and subtasks independently so
while infering rules for some path P1 for some subtask s1 then only look at the experimental
data of that path P1 and subtask s1 and infer rules and patterns from that bcz observations
of P2 doesn’t affect P1 and neither do observations for P1 but related to s2 affect P1 for s1.
So you should know that same path can be used for different subtasks and can have different
activation rules for different subtasks so while inferring these rules you should see that you
compare the object conditions for the same subtask and same path and then reach a final
conclusion like example you have some path p1 which is used in subtasks s1 and s2
and based on observations there are multiple failure cases for p1 where object size is small
and subtask is s1 while there are some success cases for same p1 where object size is big and
subtask is s2 so if you combine them you won’t be infer any rule bcz nothing’s epcific but
you need to treat both the p1’s independently one is p1 with s1 and another is p1 with s2 and
so for p1 with s1 you can infer a rule that it oonly works when object size is not small.
The output format for each path for which you can infer some rule/s will be following:
Path: [path1]
Subtask: [subtask]
Activation Rules:
* Rule a1 with some explanatory example if needed
* Rule a2 with some explanatory example if needed
.
* Rule aN with some explanatory example if needed
Path: [path2]
Subtask: [subtask]
Activation Rules:
* Rule b1 with some explanatory example if needed
* Rule b2 with some explanatory example if needed
.
* Rule bN with some explanatory example if needed"""

    user_prompt = (
        "CURRENT_RULE_TABLE:\n" + json.dumps(rules, indent=2) +
        "\n\nRECENT_TRACES:\n"   + json.dumps(traces, indent=2) +
        "\n\n(Use the template from Appendix O to infer reusable sub‑routines.)"
    )
    # ---------------------------------------------------------------------------

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user",  "content":user_prompt}],
        temperature=0.2,
    )
    try:
        proposal = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("Induction LLM returned invalid JSON – skipped.")
        TRACE_BUF.unlink(missing_ok=True)
        return False

    changed = False
    for entry in proposal.get("add", []):
        key = "SR_" + uuid.uuid5(uuid.NAMESPACE_URL,
                                 "-".join(entry["tools"])).hex[:6]
        if key in rules:                       # identical tools → already learnt
            continue
        rules[key] = {
            "subtask": entry["subtask"],
            "tools":   entry["tools"],
            "C":       round(float(entry["C"]), 3),
            "Q":       round(float(entry["Q"]), 3),
            "stamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        changed = True

    if changed:
        _write_json(CUSTOM_SRT, rules)
        print(f"Sub‑routine table updated – now {len(rules)} entries.")
    else:
        print("No beneficial new sub‑routines found.")

    TRACE_BUF.unlink(missing_ok=True)          # start a fresh buffer
    return changed