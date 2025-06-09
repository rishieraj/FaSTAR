from math import inf
import json
from re import escape, compile
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from astar_search import a_star_search, compute_quality_dynamic, extract_metadata_from_node
from inductive_reasoning import append_trace, _load_rules

def build_low_graph_for_subtask(subtask_key: str,
                                global_graph_path: str = "Tool_graph.json"):
    """
    Return {node: [successors]} restricted to *one* subtask.
    A node keeps an edge only if *both* ends end with the same subtask suffix.
    """
    with open(global_graph_path) as f:
        g = json.load(f)

    patt = compile(fr"{escape(subtask_key)}.*\(\d+\)\)$")

    low   = defaultdict(list)
    valid = {n for n in g if patt.search(n)} | {"Input Image"}

    # NEW ↓  keep *one* parent that connects Input Image with the first match
    parents = [v for v in g["Input Image"] if any(v in g and v2 in valid
                                                  for v2 in g[v])]
    valid |= set(parents)

    for u in valid:
        low[u] = [v for v in g.get(u, []) if v in valid]

    if not low["Input Image"]:
        # still isolated? fall back to direct children of Input Image
        low["Input Image"] = [v for v in g["Input Image"] if v in valid]

    return low

def run_subsequence(subsequence_map: Dict[str, List[str]],
                    alpha: float,
                    quality_threshold: float,
                    original_inputs: Dict[str, Any],
                    task_prompt: str,
                    pipeline):
    """
    Run the subsequence tree to generate the final image.
    
    Args:
        subsequence_tree (dict): The subsequence tree to run.
        alpha (float): The alpha value for A* search.
        quality_threshold (float): The quality threshold for A* search.
        original_inputs (dict): The original inputs for the pipeline.
        prompt_text (str): The prompt text for the pipeline.
        pipeline (ToolPipeline): The pipeline to use for processing.
    
    Returns:
        tuple: (full_path_taken, final_state) where `full_path_taken` is a list of (tool_name | 'A*', subtask, status_string).
    """
    SRT = {
        "SR1": {"subtask": "Object Recoloration", "tools": ["GroundingDINO", "SAM", "StabilityInpaint"], "C": 10.39, "Q": 0.89},
        "SR2": {"subtask": "Object Recoloration", "tools": ["StabilitySearchRecolor"], "C": 12.92, "Q": 0.95},
        "SR3": {"subtask": "Object Recoloration", "tools": ["YOLOv7", "SAM", "StabilityInpaint"], "C": 10.36, "Q": 0.88},
        "SR4": {"subtask": "Object Replacement", "tools": ["GroundingDINO", "SAM", "StabilityInpaint"], "C": 10.41, "Q": 0.91},
        "SR5": {"subtask": "Object Replacement", "tools": ["StabilitySearchReplace"], "C": 12.12, "Q": 0.97},
        "SR6": {"subtask": "Object Replacement", "tools": ["YOLOv7", "SAM", "StabilityInpaint"], "C": 10.38, "Q": 0.91},
        "SR7": {"subtask": "Object Removal", "tools": ["GroundingDINO", "SAM", "StabilityErase"], "C": 11.97, "Q": 0.98},
        "SR8": {"subtask": "Object Removal", "tools": ["YOLOv7", "SAM", "StabilityErase"], "C": 11.95, "Q": 0.98},
        "SR9": {"subtask": "Object Removal", "tools": ["GroundingDINO", "SAM", "StabilityInpaint"], "C": 10.39, "Q": 0.95},
        "SR10": { "subtask": "Object Removal", "tools": ["YOLOv7", "SAM", "StabilityInpaint"], "C": 10.37, "Q": 0.95},
        "SR11": { "subtask": "Text Removal", "tools": ["CRAFT", "EasyOCR", "DeepFont", "GPT4o_2", "StabilityErase"], "C": 17.81, "Q": 0.93},
        "SR12": { "subtask": "Text Removal", "tools": ["CRAFT", "EasyOCR", "DeepFont", "GPT4o_2", "DalleText"], "C": 17.95, "Q": 0.96},
        "SR13": { "subtask": "Text Removal", "tools": ["CRAFT", "EasyOCR", "DeepFont", "GPT4o_2", "TextRemovalPainting"], "C": 6.69, "Q": 0.95},
        "SR14": { "subtask": "Text Replacement", "tools": ["CRAFT", "EasyOCR", "DeepFont", "GPT4o_2", "StabilityErase", "TextWritingPillow1"], "C": 17.85, "Q": 0.92},
        "SR15": { "subtask": "Text Replacement", "tools": ["CRAFT", "EasyOCR", "DeepFont", "GPT4o_2", "DalleText", "TextWritingPillow1"], "C": 18.02, "Q": 0.94},
        "SR16": { "subtask": "Text Replacement", "tools": ["CRAFT", "EasyOCR", "DeepFont", "GPT4o_2", "TextRemovalPainting", "TextWritingPillow1"], "C": 6.77, "Q": 0.93}
    }
    SRT.update(_load_rules())

    subsequences = []
    local_memory = {}
    trace = {}
    path = ["Input Image"]
    subtasks_list = [key for key in subsequence_map]
    print(f"Subtasks: {subtasks_list}")
    for subs in subsequence_map.values():
        sub_cost = []
        for sub in subs:
            if sub in SRT:
                # Calculate the cost for each subtask
                # Cost = (time ** alpha) * (quality ** (2 - alpha))
                # where time and quality are from the SRT dictionary
                # and alpha is a parameter that can be adjusted
                time = SRT[sub]["C"]
                quality = SRT[sub]["Q"]
                cost = (time ** alpha) * (quality ** (2 - alpha))
                sub_cost.append(cost)

        # Find the minimum cost subtask
        # and add it to the subsequences list
        if not sub_cost:
            subsequences.append("None")
        else:
            min_cost = min(sub_cost)
            min_index = sub_cost.index(min_cost)
            subsequences.append(subs[min_index])

    print(f"Subsequences: {subsequences}")
    # Create a mapping of subsequences to their costs
    state: Dict[str,Any] = original_inputs.copy()

    for idx, sub in enumerate(subsequences):
        subtask = subtasks_list[idx]

        if sub != "None":
            tools = SRT[sub]["tools"]
            print(f"Processing subtask: {sub} with tools: {tools}")
            print(f"Subtask: {subtask}")
            path_cost = SRT[sub]["C"]
            path_quality = SRT[sub]["Q"]
            tool_check = {}
            for tk in tools:
                key = f"{tk} ({subtask})"
                print(f"Processing tool: {key}")
                meta = extract_metadata_from_node(key)
                state.update(meta)
                print(f"Tool metadata: {meta}")
                tool = pipeline.load_tool(tk)
                inputs = {k: state[k] for k in pipeline._get_tool_input_spec(tk)}
                result = tool.process(**inputs)
                print(f"Tool result: {result}")
                local_memory[key] = result
                path.append(key)

                if isinstance(result, dict):
                    output_image = result.get("image", state.get("image"))
                    state.update(result)
                else:
                    output_image = result
                    state['image'] = result

                quality = compute_quality_dynamic(
                    key,
                    tk,
                    original_inputs["image"],
                    output_image,
                    task_prompt,
                    result,
                    state.get("bounding_boxes", None),
                    state,
                    path,
                    local_memory
                )

                print(f"Computed quality for {key}: {quality} (Threshold: {quality_threshold})")
                if quality < quality_threshold:
                    tool_check[tk] = "fail"
                    glow = build_low_graph_for_subtask(subtask)
                    print(glow)
                    low_path, state, low_local_memory = a_star_search(
                            glow, alpha, quality_threshold,
                            state, task_prompt, pipeline)
                    print(f"Low path found: {low_path}")
                    break
                else:
                    tool_check[tk] = "success"

        else:
            glow = build_low_graph_for_subtask(subtask)
            print("Tool sub-graph for Subtask: ", glow)
            low_path, state, low_local_memory = a_star_search(
                    glow, alpha, quality_threshold,
                    state, task_prompt, pipeline)
            print("A-Star path for Subtask: ", low_path)


        if low_path:
            path.extend(low_path[1:])
            local_memory.update(low_local_memory)

        print(f"Final Path after subtask {subtask}: {path}")

        trace[sub] = {
            "subtask": subtask,
            # "tools": tools,
            # "path_cost": path_cost,
            # "path_quality": path_quality,
            # "failures": tool_check if tool_check else "success"
        }

    append_trace(trace)
    return path, state, local_memory, trace





