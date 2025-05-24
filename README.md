# FaSTA*: Fast-Slow Toolpath Agent with Subroutine Mining for Efficient Multi-turn Image Editing
📌 *This repository is under construction. Some subtasks/tools are not fully supported yet.*

---

## **Installation**  
### **1. Clone the Repository**  
```bash
git clone https://github.com/rishieraj/FaSTAR.git
cd FaSTAR  
```

### **2. Install Dependencies**  
Create a conda environment with Python 3.10 as per the following commands. Then install the dependencies from the `requirements.txt` file. 
```bash
conda create -n fastar python=3.10
conda activate fastar
pip install -r requirements.txt  
```

### **3. Download Pre-trained Checkpoints**  
The required pre-trained model checkpoints must be downloaded from **Google Drive** and placed in the `checkpoints/` folder. Please use this link to download the [checkpoints](https://drive.google.com/file/d/1E4bWTIweB_XDFeDz7OWsELC7zUZeGTtt/view?usp=share_link).

---

## **Usage**
*Note: The API keys for OpenAI and StabilityAI need to be set in the high_level.py file before executing.*
To execute **FaSTA***, run:  
```bash 
python high_level.py --image path/to/image.png --prompt "Edit this image" --output output.json --output_image final.png --alpha 0  
``` 

Example:  
```bash 
python high_level.py --image inputs/sample.jpg --prompt "Replace the cat with a dog and expand the image" --output Tree.json --output_image final_output.png --alpha 0
```  

- `--image`: Path to input image.  
- `--prompt`: Instruction for editing.  
- `--output`: Path to save generated subtask tree.  
- `--output_image`: Path to save the final output.  
- `--alpha`: Cost-quality trade-off parameter.  

---