import os
import logging
from datetime import datetime
import torch
import cv2
import numpy as np

from vace.configs import VACE_CONFIGS
from vace.annotators.subject import SubjectAnnotator
from vace.models.wan.wan_vace import WanVace
from vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from vace.models.utils.preprocessor import VaceVideoProcessor, prepare_source
from wan.utils.utils import cache_video


class VACEIntegratedPipeline:
    def __init__(self, 
                 checkpoint_dir="pretrained_models/VACE-Wan2.1-1.3B",
                 device=None,
                 model_name="vace-1.3B"):
        """
        Initialize the VACE pipeline with model loading.
        
        Args:
            checkpoint_dir: Path to WAN model checkpoint directory
            device: Device to run on ('cuda' or 'cpu')
            model_name: Model name to use (vace-1.3B or vace-14B)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        
        self.preprocess_config = VACE_CONFIGS['image_reference']
        
        self.subject_annotator = SubjectAnnotator(
            self.preprocess_config, 
            device=self.device
        )
        
        self.config = WAN_CONFIGS[model_name]
        
        self.wan_model = WanVace(
            config=self.config,
            checkpoint_dir=checkpoint_dir,
            device_id=0 if self.device.type == 'cuda' else -1,
            rank=0
        )
        
        self.vid_proc = VaceVideoProcessor(
            downsample=tuple([x * y for x, y in zip(self.config.vae_stride, self.config.patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832, 
            min_fps=self.config.sample_fps,
            max_fps=self.config.sample_fps,
            zero_start=True,
            seq_len=75600,
            keep_last=True
        )
        
        logging.info("VACEIntegratedPipeline initialized successfully")

    def _preprocess_image(self, image_path, output_dir):
        """
        Preprocess image using SubjectAnnotator with salient mode.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save processed outputs
            
        Returns:
            tuple: (src_ref_image_path, src_mask_path)
        """
        logging.info(f"Preprocessing image: {image_path}")
        
        result = self.subject_annotator.forward(
            image=image_path,
            mode="salient",
            return_mask=True
        )
        
        src_ref_image_path = os.path.join(output_dir, "src_ref_image-image_reference.png")
        src_mask_path = os.path.join(output_dir, "src_mask-image_reference.png")
        
        cv2.imwrite(src_ref_image_path, result['image'])
        cv2.imwrite(src_mask_path, result['mask'])
        
        logging.info(f"Saved preprocessed image to {src_ref_image_path}")
        logging.info(f"Saved mask to {src_mask_path}")
        
        return src_ref_image_path, src_mask_path

    def _run_inference(self, src_ref_image_path, src_mask_path, prompt, size=(832, 480), output_dir="."):
        """
        Run WAN inference to generate video.
        
        Args:
            src_ref_image_path: Path to processed reference image
            src_mask_path: Path to processed mask
            prompt: Text prompt for generation
            size: Output video size (width, height)
            output_dir: Directory to save output video
            
        Returns:
            str: Path to generated video file
        """
        logging.info(f"Running inference with prompt: {prompt}")
        
        src_ref_images, (oh, ow) = self.vid_proc.load_image(src_ref_image_path)
        src_masks, _ = self.vid_proc.load_image(src_mask_path)
        
        src_video = [None]  # No source video for image reference
        src_mask = [src_masks]
        src_ref_images = [[src_ref_images]]
        
        frame_num = 81  # Default frame count
        src_video, src_mask, src_ref_images = prepare_source(
            src_video, src_mask, src_ref_images, frame_num, (oh, ow), self.device
        )
        
        generated_video = self.wan_model.generate(
            input_prompt=prompt,
            input_frames=src_video,
            input_masks=src_mask,
            input_ref_images=src_ref_images,
            size=size,
            frame_num=frame_num,
            context_scale=1.0,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True
        )
        
        output_video_path = os.path.join(output_dir, "generated_video.mp4")
        cache_video(
            tensor=generated_video[None],
            save_file=output_video_path,
            fps=self.config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        logging.info(f"Saved generated video to {output_video_path}")
        
        return output_video_path

    def generate_video(self, image_path, prompt, size=None, output_dir=None):
        """
        Main pipeline execution: preprocess image and generate video.
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for video generation
            size: Output video size (width, height) or string format like "832*480"
            output_dir: Output directory (auto-generated if None)
            
        Returns:
            str: Path to generated video file
        """
        if size is None:
            size = (832, 480)
        elif isinstance(size, str):
            if size in SIZE_CONFIGS:
                size = SIZE_CONFIGS[size]
            else:
                try:
                    width, height = size.split('*')
                    size = (int(width), int(height))
                except:
                    logging.warning(f"Invalid size format: {size}, using default (832, 480)")
                    size = (832, 480)
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            output_dir = os.path.join("processed", "image_reference", timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            src_ref_image_path, src_mask_path = self._preprocess_image(image_path, output_dir)
            
            video_path = self._run_inference(src_ref_image_path, src_mask_path, prompt, size, output_dir)
            
            return video_path
            
        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            raise


def generate_video_from_image(image_path, prompt, size=(832, 480), checkpoint_dir="pretrained_models/VACE-Wan2.1-1.3B", model_name="vace-1.3B"):
    """
    Convenience function to generate video from image and prompt.
    
    Args:
        image_path: Path to input image
        prompt: Text prompt for video generation
        size: Output video size (width, height) or string format like "832*480"
        checkpoint_dir: Path to model checkpoint directory
        model_name: Model name to use (vace-1.3B or vace-14B)
        
    Returns:
        str: Path to generated video file
    """
    pipeline = VACEIntegratedPipeline(checkpoint_dir=checkpoint_dir, model_name=model_name)
    return pipeline.generate_video(image_path, prompt, size)
