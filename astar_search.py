import heapq
import time
import torch
import re
import os
from collections import defaultdict, deque
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import yaml
import networkx as nx
import matplotlib.pyplot as plt

import re

def extract_metadata_from_node(node_name):
    # Extract everything inside the first parentheses (subtask name)
    start = node_name.find("(")
    end = node_name.rfind(")")
    
    if start == -1 or end == -1 or end <= start:
        return {"subtask_name": None, "from_object": None, "target_object": None}

    outer_content = node_name[start + 1:end].strip()  # e.g., "Object Segmentation (Dog)(2)"
    
    # Extract subtask name (before the first parentheses)
    subtask_name = outer_content.split("(")[0].strip()  # e.g., "Object Segmentation"

    # Extract metadata from the first inner parentheses block
    match = re.search(r'\(([^()]*)\)', outer_content)
    
    from_object, target_object = None, None
    if match:
        param_str = match.group(1).strip()
        if "->" in param_str:
            parts = param_str.split("->", 1)
            from_object = parts[0].strip()
            target_object = parts[1].strip() if parts[1].strip() != "" else None
        else:
            from_object = param_str if param_str != "" else None
            target_object = None
    
    return {
        "subtask_name": subtask_name,
        "from_object": from_object,
        "target_object": target_object
    }

# This function is under construction for few tools and returns a pre-defined value for them as a fallback
def compute_quality_dynamic(next_tool, tool_name, original_image, output_image, prompt, tool_result, bounding_box, pipeline_state, current_path, local_memory):
    """
    Compute a quality score tailored to each tool.
    This code is still under construction and is currently defined for YOLO, Grounding DINO, SAM, Stable Diffusion Inpaint (Only recoloration tasks) and Stable Diffusion Search and Recolor.
    """
    from transformers import CLIPProcessor, CLIPModel
    import os, time

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    def clip_similarity_for_candidates(image, candidate_texts, save_path=""):
#        os.makedirs(save_path, exist_ok=True)
#        timestamp = int(time.time() * 1000)
#        filename = os.path.join(save_path, f"debug_crop_{timestamp}.png")
#        try:
#            image.save(filename)
#            print(f"Saved image crop to {filename}")
#        except Exception as e:
#            print(f"Failed to save image crop: {e}")
        
        inputs = processor(text=candidate_texts, images=[image], return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)
#        print("Candidate probabilities:", probs.tolist())
        return probs[0, 4].item()
    
    tool_lower = tool_name.lower()
#    print("tool lower: ", tool_lower)
    
    def get_region_from_box(box):
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return output_image.crop(box)
        return output_image

    if bounding_box is not None:
        if isinstance(bounding_box, (list, tuple)) and len(bounding_box) == 4:
            boxes = [bounding_box]
        elif isinstance(bounding_box, list) and len(bounding_box) > 0 and isinstance(bounding_box[0], (list, tuple)):
            boxes = bounding_box
        else:
            boxes = None
    elif "bounding_boxes" in tool_result and tool_result["bounding_boxes"]:
        bbs = tool_result["bounding_boxes"]
        if isinstance(bbs[0], (list, tuple)):
            boxes = bbs
        else:
            boxes = [bbs]
    else:
        boxes = None

    if boxes is None:
        boxes = [None]

    random_candidates = ["random1", "random2", "random3"]
    orig_from = pipeline_state.get("from_object")
    orig_target = pipeline_state.get("target_object")
    
    if tool_lower in ["yolov7", "groundingdino"]:
        candidate_from = "random"
        candidate_target = orig_from
        candidate_texts = random_candidates + [candidate_from, candidate_target]
#        print("yolo here")
#        print(boxes)
#        print("yolo candidate texts")
#        print(candidate_texts)
        def compute_for_boxes(boxes):
#            print(b for b in boxes)
            sims = [clip_similarity_for_candidates(get_region_from_box(b), candidate_texts) for b in boxes]
#            print("yolo q")
#            print(sims)
            return sum(sims)/len(sims) if sims else clip_similarity_for_candidates(output_image, candidate_texts)
        return compute_for_boxes(boxes)
    
    elif tool_lower == "sam":
        masks = tool_result.get("cutout_images")
#        print("sam masks")
#        print(masks)
        candidate_from1 = "random"
        candidate_target1 = orig_from
        candidate_texts1 = random_candidates + [candidate_from1, candidate_target1]
#        print("sam candidate texts")
#        print(candidate_texts1)
        if not masks:
            return clip_similarity_for_candidates(output_image, candidate_texts1)
        if not isinstance(masks, list):
            masks = [masks]
        
        sims = [clip_similarity_for_candidates(mask, candidate_texts1) for mask in masks]
#        print("sam qualities")
#        print(sims)
        return sum(sims)/len(sims) if sims else clip_similarity_for_candidates(output_image, candidate_texts1)
    
    elif tool_lower in ["stabilityinpaint"]:
        if orig_from and orig_target and orig_target.lower().endswith(orig_from.lower()):
            candidate_target = orig_target[:-len(orig_from)].strip()
            if not candidate_target:
                candidate_target = orig_target
        else:
            candidate_target = orig_target if orig_target is not None else "random"
        candidate_texts = random_candidates + ["random", candidate_target]
#        print("candidate texts:", candidate_texts)
        
        if (boxes is None or boxes == [None]):
            pipeline_boxes = pipeline_state.get("bounding_boxes", None)
            if pipeline_boxes:
                if isinstance(pipeline_boxes, (list, tuple)) and len(pipeline_boxes) > 0:
                    if isinstance(pipeline_boxes[0], (list, tuple)):
                        boxes = pipeline_boxes
                    else:
                        boxes = [pipeline_boxes]
                else:
                    boxes = [None]
        
        def compute_for_boxes(boxes):
#            print("stabiliy boxes")
#            print(boxes)
#            print(b for b in boxes)
            sims = [clip_similarity_for_candidates(get_region_from_box(b), candidate_texts) for b in boxes]
            return sum(sims)/len(sims) if sims else clip_similarity_for_candidates(output_image, candidate_texts)
        return compute_for_boxes(boxes)

    elif tool_lower in ["stabilitysearchrecolor"]:
#        print(next_tool)
        inst_matches = re.findall(r'\((\d+)\)', next_tool)
        inst = inst_matches[-1] if inst_matches else None
        metadata = extract_metadata_from_node(next_tool)
        current_subtask = metadata.get("subtask_name")

#        print("Extracted subtask name:", current_subtask, "and instance:", inst)
        
        if orig_from and orig_target and orig_target.lower().endswith(orig_from.lower()):
            candidate_target = orig_target[:-len(orig_from)].strip()
            if not candidate_target:
                candidate_target = orig_target
        else:
            candidate_target = orig_target if orig_target is not None else "random"
        
        candidate_texts = random_candidates + ["random", candidate_target]
#        print("StableSearchRecolor candidate texts:", candidate_texts)
        
        detection_boxes = None
        if inst is not None and current_subtask is not None:
            for key, res in local_memory.items():
                if key.lower().startswith("groundingdino"):
                    key_inst_matches = re.findall(r'\((\d+)\)', key)
                    key_inst = key_inst_matches[-1] if key_inst_matches else None
#                    print("key inst: ", key_inst_matches, key_inst)
                    key_subtask1 = extract_metadata_from_node(key)

                    key_subtask = key_subtask1.get("subtask_name")
#                    print("key subtask: ", key_subtask1, key_subtask)
                    if key_inst == inst and key_subtask == current_subtask:
                        if "bounding_boxes" in res and res["bounding_boxes"]:
                            detection_boxes = res["bounding_boxes"]
                            break
        if detection_boxes is not None:
            boxes = detection_boxes
        else:
            boxes = [None]
#        print("stability recolor boxes: ", boxes)
        def average_similarity(boxes):
            sims = [clip_similarity_for_candidates(get_region_from_box(b), candidate_texts) for b in boxes]
#            print("stability recolor sims: ", sims)
            return sum(sims)/len(sims) if sims else clip_similarity_for_candidates(output_image, candidate_texts)
        
        return average_similarity(boxes)
    
    elif tool_lower in ["stabilityoutpaint", "stabilityremovebg", "dalle"]:
        return 1
    
    else:
        return 1
    
def compute_g(total_cost, quality_product, alpha):
    return (total_cost ** alpha) * ((2 - quality_product) ** (2 - alpha))

def compute_heuristics_with_alpha(tool_subgraph, benchmark_table, alpha):
    leaf_nodes = [n for n, children in tool_subgraph.items() if not children]
    parents_graph = defaultdict(list)
    for parent, children in tool_subgraph.items():
        for child in children:
            parents_graph[child].append(parent)
    cost_map, quality_map, final_map = {}, {}, {}
    for leaf in leaf_nodes:
        cost_map[leaf] = 0.0
        quality_map[leaf] = 1.0
        final_map[leaf] = 0.0
    queue = deque(leaf_nodes)
    while queue:
        child = queue.popleft()
        for parent in parents_graph[child]:
            candidate_values = []
            for possible_child in tool_subgraph[parent]:
                if possible_child not in cost_map:
                    continue
                c_val = cost_map[possible_child]
                q_val = quality_map[possible_child]
                bcost, bqual = benchmark_table.get(possible_child, (0, 1))
                candidate_cost = c_val + bcost
                candidate_quality = q_val * bqual
                candidate_final = (candidate_cost ** alpha) * ((2 - candidate_quality) ** (2 - alpha))
                candidate_values.append((candidate_cost, candidate_quality, candidate_final))
            if candidate_values:
                best_cost, best_qual, best_final = min(candidate_values, key=lambda x: x[2])
                cost_map[parent] = best_cost
                quality_map[parent] = best_qual
                final_map[parent] = best_final
                if parent not in queue:
                    queue.append(parent)
    return {node: (cost_map.get(node, 0.0), quality_map.get(node, 1.0), final_map.get(node, 0.0)) for node in tool_subgraph}

def a_star_search(Gts, alpha, quality_threshold, original_inputs, task_prompt, pipeline):
    """
    Execute an A* search to find the optimal tool execution path.
    
    Each time a tool executes, its result is stored in a local memory (local_memory)
    keyed by the tool's node name.
    """
    subtask_benchmark = {
    "StabilityInpaint": {
        "Object Replacement": (12.1, 0.97),
        "Object Removal": (12.1, 0.93),
        "Object Recoloration": (12.1, 0.89)
    },
    "GPT4o_2": {
        "Question Answering based on text": (6.2, 1.0),
        "Sentiment Analysis": (6.15, 1.0)
    },
    "TextWritingPillow2": {
        "Text Replacement": (0.038, 1.0),
        "Keyword Highlighting": (0.038, 1.0)
    }
    }
    BT = {
        "YOLOv7": {"C": 0.0062, "Q": 0.82},
        "GroundingDINO": {"C": 0.1190, "Q": 1.0},
        "SAM": {"C": 0.046, "Q": 1.0},
        "DalleImage": {"C": 14.1, "Q": 1.0},
        "DalleText": {"C": 14.2, "Q": 1.0},
        "StabilitySearchRecolor": {"C": 14.7, "Q": 1.0},
        "StabilityOutpaint": {"C": 12.7, "Q": 1.0},
        "StabilityRemoveBG": {"C": 12.5, "Q": 1.0},
        "StabilityErase": {"C": 13.8, "Q": 1.0},
        "StabilityEraseText": {"C": 13.8, "Q": 0.97},
        "Stability3": {"C": 12.9, "Q": 1.0},
        "TextRemovalPainting": {"C": 0.045, "Q": 0.2},
        "DeblurGAN": {"C": 0.85, "Q": 1.0},
        "GPT4o_1": {"C": 6.31, "Q": 1.0},
        "GoogleCloudVision": {"C": 1.2, "Q": 1.0},
        "CRAFT": {"C": 1.27, "Q": 1.0},
        "CLIP": {"C": 0.0007, "Q": 1.0},
        "DeepFont": {"C": 1.8, "Q": 1.0},
        "EasyOCR": {"C": 0.15, "Q": 1.0},
        "MagicBrush": {"C": 12.8, "Q": 1.0},
        "pix2pix": {"C": 0.7, "Q": 1.0},
        "RealESRGAN": {"C": 1.7, "Q": 1.0},
        "TextWritingPillow1": {"C": 0.038, "Q": 1.0},
        "TextRedaction": {"C": 0.041, "Q": 1.0},
        "MIDAS": {"C": 0.71, "Q": 1.0}
    }

    benchmark_table = {}
    for node in Gts:
        base_tool = node.split(" (")[0].strip()
        
        if base_tool in subtask_benchmark:
            meta = extract_metadata_from_node(node)
            subtask_name = meta.get("subtask_name")
            
            if subtask_name and (subtask_name in subtask_benchmark[base_tool]):
                cost, quality = subtask_benchmark[base_tool][subtask_name]
            else:
                fallback_data = BT.get(base_tool, {"C": 0.0, "Q": 1.0})
                cost, quality = fallback_data["C"], fallback_data["Q"]
        else:
            fallback_data = BT.get(base_tool, {"C": 0.0, "Q": 1.0})
            cost, quality = fallback_data["C"], fallback_data["Q"]
        
        benchmark_table[node] = (cost, quality)
#        base_tool = node.split(" (")[0].strip()
#        bt_data = BT.get(base_tool, {"C": 0.0, "Q": 1.0})
#        benchmark_table[node] = (bt_data["C"], bt_data["Q"])

        
#    print("Final BT:")
#    print(benchmark_table)

    heuristics = compute_heuristics_with_alpha(Gts, benchmark_table, alpha)
#    print("========================================")
#    print("Heuristics")
#    print(heuristics)
#    print("========================================")

    root = "Input Image"
    init_cost = 0.0
    init_quality = 1.0
    g_root = 0
#    print(benchmark_table)
    initial_f = g_root + heuristics[root][2]
#    print(heuristics)
    queue = []
    heapq.heappush(queue, (initial_f, init_cost, init_quality, [root], original_inputs.copy()))
    best = {root: g_root}
#    print(queue, (initial_f, init_cost, init_quality, [root], original_inputs.copy()))
    
    local_memory = {}

    while queue:
        f, curr_cost, curr_quality, path, current_state = heapq.heappop(queue)
#        print(f, curr_cost, curr_quality, path, current_state)
        current_node = path[-1]
#        print("current node")
#        print(current_node)
        if not Gts.get(current_node):
            return path, current_state, local_memory
        for next_tool in Gts[current_node]:
            # Extract the base tool name (everything before " (")
            base_tool = next_tool.split(" (")[0]
            try:
                tool = pipeline.load_tool(base_tool)
            except Exception as e:
                print(f"Error loading tool {base_tool}: {e}")
                continue
#            print("next")
#            print(base_tool)

            input_spec = pipeline._get_tool_input_spec(base_tool)
            tool_inputs = {key: current_state.get(key, None) for key in input_spec}
            metadata = extract_metadata_from_node(next_tool)
            tool_inputs["subtask_name"] = metadata.get("subtask_name")
            tool_inputs["from_object"] = metadata.get("from_object")
            tool_inputs["target_object"] = metadata.get("target_object")
#            print("metadata")
#            print(metadata)
            branch_state = current_state.copy()
            branch_state["subtask_name"] = metadata.get("subtask_name")
            branch_state["from_object"]= metadata.get("from_object")
            branch_state["target_object"]= metadata.get("target_object")

            missing = [inp for inp in input_spec if inp not in tool_inputs or tool_inputs[inp] is None]
            if missing:
                print(f"Missing inputs {missing} for tool {base_tool}; skipping.")
                continue
            tool_inputs = {k: v for k, v in tool_inputs.items() if k in input_spec}
            result = tool.process(**tool_inputs)
            exec_time = result.get("execution_time", 0)
#            print("time")
#            print(exec_time)

            local_memory[next_tool] = result
#            print("result")
#            print(result)
#            print("local")
#            print(local_memory)

            if isinstance(result, dict):
                output_image = result.get("image", branch_state.get("image"))
                branch_state.update(result)
            else:
                output_image = result
                branch_state["image"] = output_image
#            print("branch_state")
#            print(branch_state)

            quality = compute_quality_dynamic(
                next_tool,
                base_tool,
                original_inputs["image"],
                output_image,
                task_prompt,
                result,
                branch_state.get("bounding_boxes", None),
                branch_state,
                path,
                local_memory
            )
#            print("quality")
#            print(quality)
            new_cost = curr_cost + exec_time
            new_quality = curr_quality * quality
            new_g = compute_g(new_cost, new_quality, alpha)

#            print(new_cost,new_quality,new_g)

            #Need to update this part for integrating the retry mechanism by defining hyperparameter changes for each model
            if quality < quality_threshold:
                print(f"Quality check failed for {next_tool} with quality {quality:.2f}")
                # continue

            if next_tool not in best or new_g < best[next_tool]:
                best[next_tool] = new_g
                f_next = new_g + heuristics.get(next_tool, (0, 1, 0))[2]
                new_path = path + [next_tool]
                heapq.heappush(queue, (f_next, new_cost, new_quality, new_path, branch_state))
    return None, None, None

##Sample Gts for the main function
#Gts = {
#    "Input Image": ['YOLOv7 (Object Replacement (dog -> cat)(1))','GroundingDINO (Object Replacement (dog -> cat)(1))' ],
#    "YOLOv7 (Object Replacement (dog -> cat)(1))": ['SAM (Object Replacement (dog -> cat)(1))'],
#    "GroundingDINO (Object Replacement (dog -> cat)(1))": ['SAM (Object Replacement (dog -> cat)(1))'],
#    "SAM (Object Replacement (dog -> cat)(1))":[]
#}

#if __name__ == "__main__":
#    alpha = 0
#    quality_threshold = 0.8
#
#    input_image_path = ""
#    input_image = Image.open(input_image_path)
#    original_inputs = {"image": input_image}
#    task_prompt = "Remove the background in the image"
#
#    from main import ToolPipeline
#    pipeline = ToolPipeline("configs/tools.yaml", auto_install=True)
#    
#    optimal_path, final_state, local_memory1 = a_star_search(
#        Gts,
#        alpha,
#        quality_threshold,
#        original_inputs,
#        task_prompt,
#        pipeline
#    )
#    
#    print("Optimal path:", optimal_path)
#    final_image = final_state.get("image")
#    if(final_image):
#        output_path = ""
#        final_image.save(output_path)
#        print(f"saved at {output_path}")
#    else:
#        print("not saved")
#        
#    print("Local Memory Final")
#    print(local_memory1)
