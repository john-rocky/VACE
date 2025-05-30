import os
import torch
import argparse
from typing import Optional, Union, Dict, Any, List

from vace.vace_preproccess import main as preprocess_main
from vace.vace_wan_inference import main as inference_main
from vace.configs import VACE_PREPROCCESS_CONFIGS


class VACEWrapper:
    """
    Python wrapper for VACE repository that provides direct function calls
    for video generation without using subprocess.
    """
    
    def __init__(self, 
                 model_name: str = "vace-1.3B",
                 ckpt_dir: str = "models/Wan2.1-VACE-1.3B/",
                 device: Optional[str] = None,
                 **kwargs):
        """
        Initialize VACE wrapper.
        
        Args:
            model_name: Model name (e.g., "vace-1.3B", "vace-14B")
            ckpt_dir: Path to model checkpoints
            device: CUDA device (auto-detected if None)
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir
        self.device = device or f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.kwargs = kwargs
        
        self.default_inference_params = {
            'size': '480p',
            'frame_num': 81,
            'sample_steps': 50,
            'sample_shift': 16,
            'sample_guide_scale': 5.0,
            'base_seed': 2025,
            'sample_solver': 'unipc',
            'use_prompt_extend': 'plain',
            'offload_model': True
        }
        
    def create_simple_video(self, 
                          reference_image: str, 
                          prompt: str,
                          output_dir: Optional[str] = None,
                          **kwargs) -> Optional[str]:
        """
        Create a simple I2V (Image-to-Video) using reference image and prompt.
        
        Args:
            reference_image: Path to reference image
            prompt: Text prompt for video generation
            output_dir: Output directory (auto-generated if None)
            **kwargs: Additional parameters for inference
            
        Returns:
            Path to generated video file
        """
        self._validate_inputs(reference_image_path=reference_image)
        
        preprocess_args = {
            'task': 'image_reference',
            'image': reference_image,
            'mode': 'salient',
            'video': None,
            'mask': None,
            'bbox': None,
            'label': None,
            'caption': None,
            'direction': None,
            'expand_ratio': None,
            'expand_num': None,
            'maskaug_mode': None,
            'maskaug_ratio': None,
            'pre_save_dir': output_dir,
            'save_fps': 16
        }
        
        preprocess_output = preprocess_main(preprocess_args)
        
        inference_args = {
            'model_name': self.model_name,
            'ckpt_dir': self.ckpt_dir,
            'prompt': prompt,
            'save_dir': output_dir,
            **self.default_inference_params,
            **kwargs
        }
        
        inference_args.update(preprocess_output)
        
        inference_output = inference_main(inference_args)
        
        return inference_output.get('out_video')
    
    def create_referenced_video(self,
                              reference_image: str,
                              reference_video: str, 
                              prompt: str,
                              mode: str = "masktrack,salient",
                              output_dir: Optional[str] = None,
                              **kwargs) -> Optional[str]:
        """
        Create a video using swap_anything with reference image and video.
        
        Args:
            reference_image: Path to reference image
            reference_video: Path to reference video
            prompt: Text prompt for video generation
            mode: Processing mode (e.g., "masktrack,salient")
            output_dir: Output directory (optional)
            **kwargs: Additional parameters for inference
            
        Returns:
            Path to generated video file
        """
        self._validate_inputs(
            reference_image_path=reference_image,
            reference_video_path=reference_video
        )
        
        preprocess_args = {
            'task': 'swap_anything',
            'image': reference_image,
            'video': reference_video,
            'mode': mode,
            'mask': None,
            'bbox': None,
            'label': None,
            'caption': None,
            'direction': None,
            'expand_ratio': None,
            'expand_num': None,
            'maskaug_mode': None,
            'maskaug_ratio': None,
            'pre_save_dir': output_dir,
            'save_fps': 16
        }
        
        preprocess_output = preprocess_main(preprocess_args)
        
        inference_args = {
            'model_name': self.model_name,
            'ckpt_dir': self.ckpt_dir,
            'prompt': prompt,
            'save_dir': output_dir,
            **self.default_inference_params,
            **kwargs
        }
        
        inference_args.update(preprocess_output)
        
        inference_output = inference_main(inference_args)
        
        return inference_output.get('out_video')
    
    def _validate_inputs(self, **kwargs):
        """Validate input parameters."""
        for key, value in kwargs.items():
            if key.endswith('_path') and value and not os.path.exists(value):
                raise FileNotFoundError(f"File not found: {value}")


def main():
    """Example usage of VACEWrapper."""
    wrapper = VACEWrapper(
        model_name="vace-1.3B",
        ckpt_dir="models/Wan2.1-VACE-1.3B/"
    )
    
    try:
        video_path = wrapper.create_simple_video(
            reference_image="assets/images/girl.png",
            prompt="A beautiful girl walking in a garden",
            frame_num=81,
            size="480p"
        )
        print(f"Generated simple video: {video_path}")
    except Exception as e:
        print(f"Error in simple video generation: {e}")
    
    try:
        video_path = wrapper.create_referenced_video(
            reference_image="assets/images/snake.png",
            reference_video="assets/videos/sample.mp4",
            prompt="A snake moving gracefully in the scene",
            mode="masktrack,salient"
        )
        print(f"Generated referenced video: {video_path}")
    except Exception as e:
        print(f"Error in referenced video generation: {e}")


if __name__ == "__main__":
    main()
