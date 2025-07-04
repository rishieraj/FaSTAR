import os
import json
import argparse
from PIL import Image
from subtask_tree import generate_subtask_tree
from subsequence import generate_subsequences
from inductive_reasoning import refine_rules_if_needed
from tool_subgraph import build_tool_subgraph_from_subtask_tree
from execute_sequence import run_subsequence
from main import ToolPipeline

def load_subtask_tree(tree_file):
    with open(tree_file, "r") as file:
        return json.load(file)
    
# Function to load subsequences from a file
def load_subsequences(subsequence_file):
    with open(subsequence_file, "r") as file:
        return json.load(file)


def main(image_path, prompt_text, output_tree="Tree.json", sequence_file="Sub.json", output_image="final_output.png", alpha=0, quality_threshold=0.8):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = Image.open(image_path)

    trace = {}
    for key in range(1, 17):
        trace[f"SR{key}"] = {
            "subtask": f"Subtask {key}",
            "tools": [],
            "path_cost": 0,
            "path_quality": 0,
            "failures": {}
        }

    # Replace 'openai_api' with the actual API key for openAI.
    os.environ['OPENAI_API_KEY'] = ''
    
    # Get API key from environment variable
    llm_api_key = os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        raise ValueError("API key for OpenAI is required. Set it as an environment variable: OPENAI_API_KEY. Ensure you have access to openAI o1 model.")
    
    subtask_tree_final = generate_subtask_tree(llm_api_key, image_path, prompt_text)
    
    with open(output_tree, "w") as f:
        json.dump(subtask_tree_final, f, indent=4)
    
    print(f"Subtask tree saved to {output_tree}")
    
    subtask_graph = load_subtask_tree(output_tree)
    final_graph = build_tool_subgraph_from_subtask_tree(subtask_graph)
    high_level_subsequences = generate_subsequences(llm_api_key, output_tree, image_path, prompt_text, trace)

    tool_subgraph_file = "Tool_graph.json"
    with open(tool_subgraph_file, "w") as f:
        json.dump(final_graph, f, indent=4)

    print(f"Tool subgraph saved to {tool_subgraph_file}")

    sequence_file = "Sub.json"
    with open(sequence_file, "w") as f:
        json.dump(high_level_subsequences, f, indent=4)

    print(f"Subsequence saved to {sequence_file}")

    # Load subsequences from the file
    subsequence_tree = load_subsequences(sequence_file)
    
    # print("=== Final Tool Subgraph ===")
    # for key, value in final_graph.items():
    #     print(f"{key}: {value}")
    
    # Replace 'stability_api' with the actual API key for StabilityAI in order to run Stable Diffusion Models.
    os.environ['STABILITY_API_KEY'] = ''
    
    # Initialize image processing pipeline
    pipeline = ToolPipeline("configs/tools.yaml", auto_install=True)
    
    original_inputs = {"image": img}

    optimal_path, final_state, local_memory, trace = run_subsequence(subsequence_tree, alpha, quality_threshold, original_inputs, prompt_text, pipeline)

    refine_rules_if_needed(llm_api_key)
    
    print("Optimal path:", optimal_path)
    
    final_image = final_state.get("image") if final_state else None
    if final_image:
        final_image.save(output_image)
        print(f"Final output saved at {output_image}")
    else:
        print("No final image generated.")
    
#    for node in optimal_path[1:]:
#        output = local_memory.get(node)
#        print(f"\nOutput for node {node}:")
#        if isinstance(output, dict):
#            img = output.get("image")
#            if img:
#                img.show()
#                print("\n")
#            else:
#                print("No image found in output.")
#        elif isinstance(output, Image.Image):
#            output.show()
#            print("\n")
#        else:
#            print("Output:", output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subtask tree and execute algorithm.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the task.")
    parser.add_argument("--output", type=str, default="Tree.json", help="Output file for the subtask tree JSON.")
    parser.add_argument("--output_image", type=str, default="final_output.png", help="Path to save the final output image.")
    parser.add_argument("--alpha", type=float, default=0, help="Alpha parameter for A* search.")
    parser.add_argument("--quality_threshold", type=float, default=0.8, help="Quality threshold for A* search.")
    
    args = parser.parse_args()
    
    # main(args.image, args.prompt, args.output, args.output_image, args.alpha, args.quality_threshold)
    main(image_path=args.image,
     prompt_text=args.prompt,
     output_tree=args.output,
     output_image=args.output_image,
     alpha=args.alpha,
     quality_threshold=args.quality_threshold)

