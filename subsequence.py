import json
import base64
from openai import OpenAI

def generate_subsequences(llm_api_key, subtask_tree, image_path, prompt):
    client = OpenAI(api_key=llm_api_key)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"

    def load_tree(path: str):
        """Return {node: [parents]} from the JSON file."""
        with open(path, "r", encoding="utf‑8") as fp:
            data = json.load(fp)
        return {item["subtask"]: item["parent"] for item in data["subtask_tree"]}
    
    def build_paths(graph):
        """Return every root‑to‑leaf path in the DAG as a list of lists."""
        roots = [n for n, parents in graph.items() if not parents]

        paths = []

        def dfs(node, trail):
            trail.append(node)
            # children = nodes that list *this* node as a parent
            children = [n for n, parents in graph.items() if node in parents]
            if not children:                       # leaf
                paths.append(trail.copy())
            else:
                for c in children:
                    dfs(c, trail)
            trail.pop()

        for r in roots:
            dfs(r, [])
        return paths
    
    def arrows(trail):
        """Convert a list like [A, B, C] to 'A -> B -> C'."""
        return " -> ".join(trail)
    
    tree_list = []
    graph = load_tree(subtask_tree)
    for trail in build_paths(graph):
        tree_list.append(arrows(trail))

    tree_text = "\n".join(f"{i+1}. {path}" for i, path in enumerate(tree_list))

    message = f"""So we have this image and also have the following input prompt:

{prompt}

So we got the following subtask tree:

\n\n{tree_text}\n\n

Note that in the subtask tree within a particular node there is a bracket which tell us about the object from and target and if there is only from then target is not mentioned like in removal only from object is needed no target is required while for replacement/recoloration target is also required


Now we have the following subsequences list for each subtask and each of the subsequences have some observations related to them which specify under which conditions they are to be used or not. So you need to read those subsequences and their observations then check the corresponding object for that subtask within the image like if its  Object Removal (Cat) then check the cat in image and then from the subsequences list check that if for that particular subtask there is any subsequence in which the observation conditions are satisfied and if so give the list of those subsequences for that subtask and you need to do this for all subtasks in the subtask tree.


Subsequence list and the details:

Subtask	            Subsequence	Path	Estimated Cost	Estimated Quality	Info learned from the past experiences
Object Recoloration	SR1 				This is preferred if the object in play isn’t supported as a class by YOLO and the object is not too small and also there is no extra information like text, some object, etc in front or on the object which is an essential object or which may be required in some other subsequent subtask like some text written on a car which is to be recolored and this text is to be operated upon in subsequent subtasks maybe there is some text replacement, removal, etc for this particular text ahead in the subtask tree. We will have to assume that by using this subsequence the entire object will be changed completely so any other object or text which overlaps with it will also change so if that object is important or maybe to be used in later subtasks then this subsequence should not be used and we will not use this also in cases where the object to recolor is quite small.
Object Recoloration	SR2 				This is used in cases where the difference between the initial and target colors of the object is not too much like if it is currently a white object and need to change to black or dark blue, etc then this subsequence will not be preferred.
Object Recoloration	SR3				    This is preferred only if the object in play is supported as a class by YOLO and the object is not too small and also there is no extra information like text, some object, etc in front or on the object which is an essential object or which may be required in some other subsequent subtask like some text written on a car which is to be recolored and this text is to be operated upon in subsequent subtasks maybe there is some text replacement, removal, etc for this particular text ahead in the subtask tree. We will have to assume that by using this subsequence the entire object will be changed completely so any other object or text which overlaps with it will also change so if that object is important or maybe to be used in later subtasks then this subsequence should not be used and we will not use this also in cases where the object to recolor is quite small.
Object Replacement	SR4				    This is preferred if the object in play isn’t supported as a class by YOLO and the object is not too small. This will not be preferred in cases where the size difference between the initial object and target object is too big like replacing a hen with a car or the shape difference between them is confusing like replacing a bench with chair which are very similar objects so the model might confuse. It works fine if the size difference is not too much like hen and child or hen and dog, etc.
Object Replacement	SR5			        This subsequence is not preferable if there are multiple instances present of the object which is to be replaced like multiple cats present which are to be replaced with dogs and also it is not preferable when the object to be replaced is not very common or not properly visible like maybe a bit transparent or looking very thin or if the object is cut like maybe only half the car is visible, etc. or if the target object is very very different in size or shape like cycle to car and it wont be easy to fit the new object in that space
Object Replacement	SR6				    This is preferred if the object in play is supported as a class by YOLO and the object is not too small. This will not be preferred in cases where the size difference between the initial object and target object is too big like replacing a hen with a car or the shape difference between them is confusing like replacing a bench with chair which are very similar objects so the model might confuse. It works fine if the size difference is not too much like hen and child or hen and dog, etc.
Object Removal	    SR7			        This is preferred if the object to be removed is not supported as a class by YOLO. This is preferred if the background behind object to be removed is not complex with a lot of different objects and when the object is not too big. It is not used in cases where there are lots of different objects in background or if the object in background has specific shape like a person or some object which needs to be drawn after removal of the foreground object but if there are minor objects in background or the objects present can be filled like a wall or ground, etc then its fine but if there are lots of objects and some specific detailed objects with specific shapes and all then it is not preferred or also in cases when the background object is something specific like a person or something which the current object is occluding and which needs to be completed and drawn and not just filled if this object is removed if the background object is just some wall or ground or something like that then it can be filled
Object Removal	    SR8			        This is preferred if the object to be removed is supported as a class by YOLO. This is preferred if the background behind object to be removed is not complex with a lot of different objects and when the object is not too big. It is not used in cases where there are lots of different objects in background or if the object in background has specific shape like a person or some object which needs to be drawn after removal of the foreground object but if there are minor objects in background or the objects present can be filled like a wall or ground, etc then its fine but if there are lots of objects and some specific detailed objects with specific shapes and all then it is not preferred or also in cases when the background object is something specific like a person or something which the current object is occluding and which needs to be completed and drawn and not just filled if this object is removed if the background object is just some wall or ground or something like that then it can be filled
Object Removal	    SR9				    This is preferred if the object to be removed is not supported as a class by YOLO. This is preferred if the background behind object to be removed is not clear and there are objects in background which the current object occludes and which require drawing to complete them after the removal of current object to look real and also in cases when the object to be removed is very big. It isn’t preferred when the object is not big and has a plain or clean background where after removal only background needs to be extended and no object needs to be added.
Object Removal	    SR10			    This is preferred if the object to be removed is supported as a class by YOLO. This is preferred if the background behind object to be removed is not clear and there are objects in background which the current object occludes and which require drawing to complete them after the removal of current object to look real and also in cases when the object to be removed is very big. It isn’t preferred when the object is not big and has a plain or clean background where after removal only background needs to be extended and no object needs to be added.
Text Removal	    SR11				Isn’t preferred if there is something that needs to be added in background in place of the text removed to make it look real like if there were some clouds, or text was written on top of an image with lots of details like kids, etc maybe like a watermark but works well if the background is plain and simple or even simple textured without specific objects
Text Removal	    SR12				Works fine if the background behind text has some simple objects like clouds, etc where adding some extra artifact won’t make a difference but isn’t preferred where the surrounding objects are similar like have text or something so that would increases chances that some extra artifacts like text is added to that place
Text Removal	    SR13				This works only in cases where the background where text is written is absolutely plain and has solid color and if the box within which text is present replaced with a solid color it would blend with the background well without any issue but if the background where text is written has any objects or simple texture or gradient based fill then it wont be preferred
Text Replacement	SR14				Isn’t preferred if there is something that needs to be added in background in place of the text removed to make it look real like if there were some clouds, or text was written on top of an image with lots of details like kids, etc maybe like a watermark but works well if the background is plain and simple or even simple textured without specific objects. These conditions apply for the text removal part before the writing is done in that place.
Text Replacement	SR15				Works fine if the background behind text has some simple objects like clouds, etc where adding some extra artifact won’t make a difference or maybe is needed like a busy street or an image with a lot of objects and text is written on top of that maybe like a watermark or something like that but isn’t preferred where the surrounding objects are similar like have text or something so that would increases chances that some extra artifacts like text is added to that place or in plain backgrounds
Text Replacement	SR16				This works only in cases where the background where text is written is absolutely plain and has solid color and if the box within which text is present replaced with a solid color it would blend with the background well without any issue but if the background where text is written has any objects or simple texture or gradient based fill then it wont be preferred


Example:

Suppose we have an image which has lots of objects along with a very large car which has a background with lots of objects and also a brown wooden board with some text written on it. Now we have a prompt that remove the car and recolor the wooden board to pink and detect the text and get the following subtask tree:

Object Removal (Car) -> Object Recoloration (Wooden Board -> Pink Wooden Board) -> Text Detection ()

Now we see the subsequence list and find that for removal since the object is too big sub7 and sub8 are not possible. Now in sub9 and sub10 we see that the 'car' class is supported by yolo so eventually we choose sub10 for this subtask. For recoloration we see that it has object (text) which is imp and is involved in subsequent subtask so sub1 and sub3 aren't possible and we see that the color of board is light brown so light brown and pink dont have too much difference so we choose sub2. For text detection there is not subsequence available so we leave it like that

So output will be:
Object Removal (Car) : [SR10]
Object Recoloration (Wooden Board -> Pink Wooden Board) : [SR2]
Text Detection () : [None]


Now lets say the wooden board was black in color and had to be recolored to white. In this case the sub1 and sub3 are not possible because of the text as before but now sub2 is also not possible because the color difference is too much. So we do not choose any subsequence for this subtask and output is as follows:

Object Removal (Car) : [SR10]
Object Recoloration (Wooden Board -> Pink Wooden Board) : [None]
Text Detection () : [None]

Now lets change the details further. Lets say that the wooden board does not have any text written on it and has to be recolored from pink to yellow and the text detection subtask wasn't present so in this case for recoloration all subsequences are possible except sub3 bcz wooden board isnt a class supported by yolo

New output:

Object Removal (Car) : [SR10]
Object Recoloration (Wooden Board -> Pink Wooden Board) : [SR1, SR2]

Now lets change it a bit assume that all conditions are as original but the car is small and behind the car only some walls, grass, etc are present some basic stuff and not a lot of objects like occluded people, cats, etc so in this case we will choose sub8 and sub10 for it as it is not too plain that sub10 cannot be used and it is not way too complex that sub8 cannot be used

New output:

Object Removal (Car) : [SR8, SR10]
Object Recoloration (Wooden Board -> Pink Wooden Board) : [SR2]
Text Detection () : [None]

Now you need to do the same things for the current case where the input prompt is : Remove the dog, segment the person, estimate the depth of the scene, and remove the Background.

Subtask Tree: Object Removal (Dog)(1) -> Object Segmentation (Person)(2) -> Background Removal (3) -> Depth Estimation (Scene)(4)

One important thing to keep in mind is that you skip sub2 only if the color difference is way too huge like white to black but not when its only little difference like black to yellow or blue to yellow/pink like you need to see that the minimum difference between any shade of the two colors isnt huge like for blue to pink the lightest shade of blue and pink or the darkest shade of pink and blue arent much different in darkness while for black and white there is no combination of shades where the difference is not very huge. Also multiple options are possible if they satisfy all the conditions it is not necessary that only one is chosen and it is also possible that no subsequence fulfills all conditions so in that case choose None so that we can do A* search and find the correct output.

So you need to extract all relevant details related to all relevant objects from the image given to you then check the subsequence list if anyone matches and give the output."""

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
        model="gpt-4o",
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
