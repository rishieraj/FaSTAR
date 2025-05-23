import json
import base64
from openai import OpenAI

def generate_subtask_tree(llm_api_key, image_path, prompt):
    client = OpenAI(api_key=llm_api_key)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"

    message = f"""You are an advanced reasoning model responsible for decomposing a given image editing task into a structured subtask tree. Your task is to generate a well-formed subtask tree that logically organizes all necessary steps to fulfill the given user prompt. Below are key guidelines and expectations:

1. Understanding the Subtask Tree

A subtask tree is a structured representation of how the given image editing task should be broken down into smaller, logically ordered subtasks. Each node in the tree represents a subtask which is involved in the prompt, and edges represent the ordering like which subtask needs to be completed before or after which.

Each node of the tree represents the subtasks required to complete the task. The tree ensures that all necessary operations are logically ordered, meaning a subtask that depends on another must appear after its dependency.

2. Steps to Generate the Subtask Tree
    •    Step 1: Identify all relevant subtasks needed to fulfill the given prompt.
    •    Step 2: Ensure that each subtask is logically ordered, meaning operations dependent on another should be placed later in the path.
    •    Step 3: Each subtask should be uniquely labeled based on the object it applies to and of the format (Obj1 -> Obj2) where obj1 is to be replaced with obj2 and in case of recoloring (obj -> new color) while with removal just include (obj) which is to be removed. Example: If two objects require replacement, the subtasks should be labeled distinctly, such as Object Replacement (Obj1 -> Obj2).
    •    Step 4: A tree may involve multiple correct paths like subtask1 and subtask2 are not related on each other so one path can be subtask1->subtask2 and another can be subtask2->subtask1. For such cases the occurrence of these subtasks are repeated. eg. here subtask1 is appearing twice once as a parent node once as a child node. In such cases, you should number them like suubtask1(1) and subtask1(2) and subtask2(1) and subtask2(2) so now it will look like subtask1(1)->subtask2(2), subtask2(1)->subtask1(2)
    •    Step 5: There also might be multiple possible subtasks for a particular requirement like if a part of task is to replace the cat with a pink dog then the two possible ways are Object Replacement (cat-> pink dog) and another is Object Replacement (cat->dog) -> Object Recoloration (dog->pink)

3. Logical Constraints & Dependencies

When constructing the tree, keep in mind that you take care of the order as well like if a task involves replacing an object with something and then doing some operation on the new object then this operation should always be after the object replacement for this object since we cannot do the operation on the new object till it is actually created and in the image.

4. Input Format

The LLM will receive:
    1.    An image.
    2.    A text prompt describing the editing task.
    3.    A predefined list of subtasks the model supports (provided below).

Supported Subtasks

Here is the complete list of subtasks available for constructing the subtask tree:
Object Detection
Object Segmentation
Object Addition
Object Removal
Background Removal
Landmark Detection
Object Replacement
Image Upscaling
Image Captioning
Changing Scenery
Object Recoloration
Outpainting
Depth Estimation
Image Deblurring
Text Extraction
Text Replacement
Text Removal
Text Addition
Text Redaction
VQuestion Answering based on text
Keyword Highlighting
Sentiment Analysis
Caption Consistency Check
Text Detection

You must strictly use only these subtasks when constructing the tree.

5. Expected Output Format

The model should output the subtask tree in structured JSON format, where each node contains:
    •    Subtask Name (with object label if applicable)
    •    Parent Node (Parent node of that subtask)
    •    Execution Order (logical flow of tasks)

6. Example Inputs & Expected Outputs

Here are some example prompts along with the expected subtask trees:

Example 1

Input Prompt:
“Detect the pedestrians, remove the car and replacement the cat with rabbit and recolor the dog to pink.”

Expected Subtask Tree:

{{
    "task": "Detect the pedestrians, remove the car and replacement the cat with rabbit and recolor the dog to pink",
    "subtask_tree": [
        {{
            "subtask": "Object Detection (Pedestrian)(1)",
            "parent": []
        }},
        {{
            "subtask": "Object Removal (Car)(2)",
            "parent": ["Object Detection (Pedestrian)(1)"]
        }},
        {{
            "subtask": "Object Replacement (Cat -> Rabbit)(3)",
            "parent": ["Object Removal (Car)(2)"]
        }},
        {{
            "subtask": "Object Replacement (Cat -> Rabbit)(4)",
            "parent": ["Object Detection (Pedestrian)(1)"]
        }},
        {{
            "subtask": "Object Removal (Car)(5)",
            "parent": ["Object Replacement (Cat -> Rabbit)(4)"]
        }},
        {{
            "subtask": "Object Recoloration (Dog -> Pink Dog)(6)",
            "parent": ["Object Replacement (Cat -> Rabbit)(3)", "Object Removal (Car)(5)"]
        }}
    ]
}}

Example 2

Input Prompt:
“Update the closed signage to open while detecting the trash can and pedestrian crossing for better scene understanding. Also, remove the people for clarity.”

Expected Subtask Tree:

{{
    "task": "Update the closed signage to open while detecting the trash can and pedestrian crossing for better scene understanding. Also, remove the people for clarity.",
    "subtask_tree": [
        {{
            "subtask": "Text Replacement (CLOSED -> OPEN)(1)",
            "parent": []
        }},
        {{
            "subtask": "Object Detection (Pedestrian Crossing)(2)",
            "parent": ["Text Replacement (CLOSED -> OPEN)(1)"]
        }},
        {{
            "subtask": "Object Detection (Trash Can)(3)",
            "parent": ["Text Replacement (CLOSED -> OPEN)(1)"]
        }},
        {{
            "subtask": "Object Detection (Pedestrian Crossing)(4)",
            "parent": ["Object Detection (Trash Can)(3)"]
        }},
        {{
            "subtask": "Object Detection (Trash Can)(5)",
            "parent": ["Object Detection (Pedestrian Crossing)(2)"]
        }},
        {{
            "subtask": "Object Removal (People)(6)",
            "parent": ["Object Detection (Pedestrian Crossing)(4)", "Object Detection (Trash Can)(5)"]
        }}     
    ]
}}

7. Your Task

Now, using the given input image and prompt, generate a well-structured subtask tree that adheres to the principles outlined above.
    •    Ensure logical ordering and clear dependencies.
    •    Label subtasks by object name where needed.
    •    Structure the output as a JSON-formatted subtask tree.

Input Details
    •    Image: input image
    •    Prompt: ["{prompt}"]
    •    Supported Subtasks: (See the list above)

Now, generate the correct subtask tree.
Before you generate the tree you need to make sure that for every path possible in the subtask tree all the subtasks in that tree are covered and none are skipped. Also if a prompt involves something related to object replacement then just have that you dont need to think about its prerequisites like detecting it or anything bcz it already covers it. Also make sure that there is only one start node so that would mean that there would be only one subtask which doesnt have any parent node. Also, things like the outpainting should be always at first because they will add more details to image which might be related to other subtasks. Also, it is not necessary that the path you choose has the same order as the order of subtasks mentioned in prompt. It can be different based on some dependencies which you might think are there and there might be multiple possible sequences as well like subtask1(1) -> subtask2(1) -> subtask3(1) and subtask1(1) -> subtask3(2) -> subtask2(2) but you need to make sure it is not very long. also if there is a subtask which is there at the exact same location in different possible paths then its number will be same like in this example subtask1 was always at the same location so its number was always same"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message + "\n\nYou must respond in a valid JSON format. Do not include any extra text before or after the JSON output."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages
    )
    
    content = response.choices[0].message.content
    if content.startswith("```json"):
        content = content[7:-3]
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("Failed to parse response:")
        print("Raw content:", content)
        raise ValueError(f"Invalid JSON response: {str(e)}") from e

    return subtask_tree
