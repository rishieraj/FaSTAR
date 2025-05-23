import argparse
import yaml
from pathlib import Path
from importlib import import_module
from PIL import Image
from typing import Dict, Any
from tools import BaseTool
import numpy as np

class ToolPipeline:
    def __init__(self, config_path: str, auto_install: bool = False):
        self.tool_configs = self._load_config(config_path)
        self.auto_install = auto_install

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path) as f:
            return yaml.safe_load(f)['tools']
        
    def load_tool(self, tool_name: str) -> BaseTool:
        tool_cfg = self.tool_configs[tool_name]
        
        if self.auto_install and 'requirements' in tool_cfg:
            from tools import DependencyManager
            DependencyManager.install_requirements(tool_cfg['requirements'])
        
        module = import_module(f"tools.{tool_name.lower()}")
        tool_class = getattr(module, tool_cfg['class'])
        
        return tool_class(tool_cfg)

    def _get_tool_input_spec(self, tool_name: str) -> Dict:
        return self.tool_configs[tool_name].get('inputs', ['image'])

    def run(self, inputs: Dict[str, Any], tool_chain: list) -> Dict[str, Any]:
        current_data = inputs.copy()
        
        for tool_name in tool_chain:
            tool = self.load_tool(tool_name)
            input_spec = self._get_tool_input_spec(tool_name)
            
            args = {k: current_data[k] for k in input_spec if k in current_data}
            
            # Execute tool and update state
            result = tool.process(**args)
            
            if isinstance(result, dict):
                current_data.update(result)
            else:
                current_data['image'] = result
                
        return current_data

#def main():
#    parser = argparse.ArgumentParser(description='Multi-Modal ML Pipeline')
#    
#    parser.add_argument('--tools', nargs='+', required=True,
#                       help='Tool execution order')
#    parser.add_argument('--config', default='configs/tools.yaml')
#    parser.add_argument('--auto-install', action='store_true')
#    
#    parser.add_argument('--image', type=str, help='Input image path')
#    parser.add_argument('--mask', type=str, help='Mask image path')
#    parser.add_argument('--prompt', type=str, help='Text prompt')
#    parser.add_argument('--output-dir', type=str, default='outputs')
#    parser.add_argument('--search_prompt', type=str, 
#                       help='Search prompt for Stability Search/Replace')
#    parser.add_argument('--select_prompt', type=str,
#                   help='Object to recolor for Stability Search/Recolor')
#    parser.add_argument('--left', type=int, help='Pixels to expand left side')
#    parser.add_argument('--down', type=int, help='Pixels to expand downward')
#    parser.add_argument('--right', type=int, help='Pixels to expand right side')
#    parser.add_argument('--up', type=int, help='Pixels to expand upward')
#    parser.add_argument('--points', type=str,
#                    help='Path to numpy file with batched points (shape: batch×num_points×2)')
#    parser.add_argument('--labels', type=str,
#                    help='Path to numpy file with batched labels (shape: batch×num_points)')
#    parser.add_argument('--edit', type=str, help='Edit instruction for MagicBrush')
#    parser.add_argument('--text-prompt', type=str, 
#                   help='Detection prompt (e.g., "chair . person . dog .")')
#    parser.add_argument('--image_path', type=str, help='Input image path')
#    parser.add_argument('--from_object', type=str, help='Object Detection')
#
#    
#    args = parser.parse_args()
#    
#    pipeline = ToolPipeline(args.config, args.auto_install)
#    
#    inputs = {}
#    for tool_name in args.tools:
#        required_inputs = pipeline._get_tool_input_spec(tool_name)
#        for inp in required_inputs:
#            if inp == 'image' and args.image:
#                inputs['image'] = Image.open(args.image)
#                inputs['image_name'] = Path(args.image).stem
#            elif inp == 'mask' and args.mask:
#                inputs['mask'] = Image.open(args.mask)
#            elif inp == 'prompt' and args.prompt:
#                inputs['prompt'] = args.prompt
#            elif inp == 'search_prompt' and args.search_prompt:
#                inputs['search_prompt'] = args.search_prompt
#            elif inp == 'select_prompt' and args.select_prompt:
#                inputs['select_prompt'] = args.select_prompt
#            # Add these new conditions
#            elif inp == 'left' and args.left is not None:
#                inputs['left'] = args.left
#            elif inp == 'right' and args.right is not None:
#                inputs['right'] = args.right
#            elif inp == 'up' and args.up is not None:
#                inputs['up'] = args.up
#            elif inp == 'down' and args.down is not None:
#                inputs['down'] = args.down
#            # In input collection section
#            elif inp == 'points' and args.points:
#                inputs['points'] = np.load(args.points)
#            elif inp == 'labels' and args.labels:
#                inputs['labels'] = np.load(args.labels)
#            elif inp == 'edit' and args.edit:
#                inputs['edit'] = args.edit
#            elif inp == 'text_prompt' and args.text_prompt:
#                inputs['text_prompt'] = args.text_prompt
#            elif inp == 'image_path' and args.image_path:
#                inputs['image_path'] = args.image_path
#            elif inp == 'from_object' and args.from_object:
#                inputs['from_object'] = args.from_object
#
#            elif inp not in inputs:
#                raise ValueError(f"Missing required input '{inp}' for tool {tool_name}")
#
#    results = pipeline.run(inputs, args.tools)
##    print(results)
#    
#    Path(args.output_dir).mkdir(exist_ok=True)
#    for name, data in results.items():
#        print(f"Key: {name}, Type: {type(data)}")
#        if isinstance(data, Image.Image):
#            output_path = Path(args.output_dir) / f"{name}.png"
#            print(f"Saving image to {output_path}")
#            data.save(output_path)
#        elif isinstance(data, str):
#            with open(Path(args.output_dir) / f"{name}.txt", 'w') as f:
#                f.write(data)
#
#if __name__ == '__main__':
#    main()
