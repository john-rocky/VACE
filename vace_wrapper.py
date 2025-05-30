import os
import subprocess
import time
import glob
from pathlib import Path
import shutil
import tempfile

class VACEWrapper:
    """
    A wrapper class for VACE video generation functionality.
    This class provides simplified methods to generate videos from static images using VACE.
    """
    
    def __init__(self, vace_dir=None, model_path=None):
        """
        Initialize the VACE wrapper.
        
        Args:
            vace_dir (str, optional): Path to the VACE repository directory.
                                     If None, assumes current directory is VACE repo.
            model_path (str, optional): Path to the VACE model checkpoint.
                                       If None, uses the default 1.3B model.
        """
        if vace_dir is None:
            self.vace_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.vace_dir = vace_dir
            
        if model_path is None:
            self.model_path = os.path.join(self.vace_dir, "models/Wan2.1-VACE-1.3B")
        else:
            self.model_path = model_path
            
        self.output_dir = os.path.join(self.vace_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._verify_vace_installation()
    
    def _verify_vace_installation(self):
        """Verify that VACE is properly installed."""
        vace_preprocess_path = os.path.join(self.vace_dir, "vace/vace_preproccess.py")
        vace_inference_path = os.path.join(self.vace_dir, "vace/vace_wan_inference.py")
        
        if not os.path.exists(vace_preprocess_path) or not os.path.exists(vace_inference_path):
            raise FileNotFoundError(f"VACE scripts not found in {self.vace_dir}. Please provide the correct VACE directory.")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please provide the correct model path.")
    
    def _run_command(self, command):
        """Run a shell command and return the output."""
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Command failed with error: {stderr}")
            raise RuntimeError(f"Command failed: {command}\nError: {stderr}")
        
        return stdout
    
    def _get_latest_processed_dir(self, task):
        """Get the most recently created processed directory for a specific task."""
        processed_dir = os.path.join(self.vace_dir, "processed", task)
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
        
        timestamp_dirs = glob.glob(os.path.join(processed_dir, "*"))
        if not timestamp_dirs:
            raise FileNotFoundError(f"No timestamp directories found in {processed_dir}")
        
        latest_dir = max(timestamp_dirs, key=os.path.getctime)
        return latest_dir
    
    def _get_processed_files(self, task):
        """Get the processed files for a specific task."""
        latest_dir = self._get_latest_processed_dir(task)
        
        src_video = glob.glob(os.path.join(latest_dir, f"src_video-{task}.mp4"))[0]
        src_mask = glob.glob(os.path.join(latest_dir, f"src_mask-{task}.mp4"))[0]
        src_ref_images = glob.glob(os.path.join(latest_dir, f"src_ref_image_*-{task}.png"))
        
        return {
            "src_video": src_video,
            "src_mask": src_mask,
            "src_ref_images": ",".join(src_ref_images) if src_ref_images else None
        }
    
    def _run_inference(self, task, prompt, output_name=None):
        """
        Run inference using the processed files.
        
        Args:
            task (str): The task name used for preprocessing.
            prompt (str): Text prompt describing the desired video.
            output_name (str, optional): Name of the output video file.
            
        Returns:
            str: Path to the generated video.
        """
        processed_files = self._get_processed_files(task)
        
        if output_name is None:
            output_name = f"{task}_{int(time.time())}.mp4"
        output_path = os.path.join(self.output_dir, output_name)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        inference_cmd = [
            f"python {os.path.join(self.vace_dir, 'vace/vace_wan_inference.py')}",
            f"--ckpt_dir {self.model_path}",
            f"--src_video {processed_files['src_video']}",
            f"--src_mask {processed_files['src_mask']}",
        ]
        
        if processed_files['src_ref_images']:
            inference_cmd.append(f"--src_ref_images {processed_files['src_ref_images']}")
        
        inference_cmd.append(f"--prompt \"{prompt}\"")
        inference_cmd.append(f"--save_file {output_path}")
        
        self._run_command(" ".join(inference_cmd))
        
        if not os.path.exists(output_path):
            results_dir = os.path.join(self.vace_dir, "results")
            if os.path.exists(results_dir):
                mp4_files = []
                for root, _, files in os.walk(results_dir):
                    for file in files:
                        if file.endswith(".mp4") and "out_video" in file:
                            file_path = os.path.join(root, file)
                            mp4_files.append((file_path, os.path.getctime(file_path)))
                
                if mp4_files:
                    latest_file = max(mp4_files, key=lambda x: x[1])[0]
                    shutil.copy(latest_file, output_path)
                    print(f"Found output video at {latest_file}, copied to {output_path}")
                else:
                    raise FileNotFoundError(f"Output video not found at {output_path} or in results directory")
            else:
                raise FileNotFoundError(f"Output video not found at {output_path} and results directory does not exist")
        
        return output_path
    
    def create_simple_video(self, image_path, prompt, output_name=None):
        """
        Create a simple image-to-video conversion using image_reference.
        
        Args:
            image_path (str): Path to the reference image.
            prompt (str): Text prompt describing the desired video.
            output_name (str, optional): Name of the output video file.
            
        Returns:
            str: Path to the generated video.
        """
        preprocess_cmd = [
            f"python {os.path.join(self.vace_dir, 'vace/vace_preproccess.py')}",
            f"--task image_reference",
            f"--mode salientmasktrack",
            f"--image {image_path}",
            f"--maskaug_mode original",
            f"--maskaug_ratio 0.1"
        ]
        
        self._run_command(" ".join(preprocess_cmd))
        
        return self._run_inference("image_reference", prompt, output_name)
    
    def create_referenced_video(self, image_path, video_path, prompt, output_name=None):
        """
        Create a video by swapping objects in a reference video with objects from an image.
        
        Args:
            image_path (str): Path to the reference image.
            video_path (str): Path to the reference video.
            prompt (str): Text prompt describing the desired video.
            output_name (str, optional): Name of the output video file.
            
        Returns:
            str: Path to the generated video.
        """
        preprocess_cmd = [
            f"python {os.path.join(self.vace_dir, 'vace/vace_preproccess.py')}",
            f"--task swap_anything",
            f"--mode label,plain",
            f"--label 'car'",
            f"--maskaug_mode original",
            f"--maskaug_ratio 0.1",
            f"--video {video_path}",
            f"--image {image_path}"
        ]
        
        self._run_command(" ".join(preprocess_cmd))
        
        return self._run_inference("swap_anything", prompt, output_name)
    
    def create_run_video(self, image_path, prompt, output_name=None):
        """
        Create a video where the object moves from right to left.
        
        Args:
            image_path (str): Path to the reference image.
            prompt (str): Text prompt describing the desired video.
            output_name (str, optional): Name of the output video file.
            
        Returns:
            str: Path to the generated video.
        """
        from PIL import Image
        img = Image.open(image_path)
        width, height = img.size
        
        right_x = int(width * 0.7)
        left_x = int(width * 0.2)
        y_center = int(height * 0.5)
        box_size = int(min(width, height) * 0.3)
        
        start_box = f"{right_x - box_size//2},{y_center - box_size//2},{right_x + box_size//2},{y_center + box_size//2}"
        end_box = f"{left_x - box_size//2},{y_center - box_size//2},{left_x + box_size//2},{y_center + box_size//2}"
        
        preprocess_cmd = [
            f"python {os.path.join(self.vace_dir, 'vace/vace_preproccess.py')}",
            f"--task move_anything",
            f"--bbox '{start_box} {end_box}'",
            f"--expand_num 80",
            f"--label 'car'",
            f"--image {image_path}",
            f"--maskaug_mode original",
            f"--maskaug_ratio 0.1"
        ]
        
        self._run_command(" ".join(preprocess_cmd))
        
        return self._run_inference("move_anything", prompt, output_name)
    
    def create_animation_video(self, image_path, prompt, output_name=None):
        """
        Create a video with natural animation of the object.
        
        Args:
            image_path (str): Path to the reference image.
            prompt (str): Text prompt describing the desired video.
            output_name (str, optional): Name of the output video file.
            
        Returns:
            str: Path to the generated video.
        """
        preprocess_cmd = [
            f"python {os.path.join(self.vace_dir, 'vace/vace_preproccess.py')}",
            f"--task animate_anything",
            f"--mode salientbboxtrack",
            f"--image {image_path}",
            f"--maskaug_mode original",
            f"--maskaug_ratio 0.1"
        ]
        
        self._run_command(" ".join(preprocess_cmd))
        
        return self._run_inference("animate_anything", prompt, output_name)
    
    def create_complex_run_video(self, image_path, prompt, output_name=None):
        """
        Create a video where the object moves in a serpentine path toward the viewer.
        
        Args:
            image_path (str): Path to the reference image.
            prompt (str): Text prompt describing the desired video.
            output_name (str, optional): Name of the output video file.
            
        Returns:
            str: Path to the generated video.
        """
        from PIL import Image
        img = Image.open(image_path)
        width, height = img.size
        
        back_size = int(min(width, height) * 0.2)
        mid_size = int(min(width, height) * 0.25)
        front_size = int(min(width, height) * 0.35)
        
        center_x = width // 2
        center_y = height // 2
        
        box1 = f"{center_x - back_size//2},{center_y - back_size//2},{center_x + back_size//2},{center_y + back_size//2}"
        box2 = f"{center_x - mid_size//2 + int(width*0.15)},{center_y - mid_size//2},{center_x + mid_size//2 + int(width*0.15)},{center_y + mid_size//2}"
        box3 = f"{center_x - mid_size//2 - int(width*0.15)},{center_y - mid_size//2},{center_x + mid_size//2 - int(width*0.15)},{center_y + mid_size//2}"
        box4 = f"{center_x - front_size//2},{center_y - front_size//2},{center_x + front_size//2},{center_y + front_size//2}"
        
        preprocess_cmd = [
            f"python {os.path.join(self.vace_dir, 'vace/vace_preproccess.py')}",
            f"--task move_anything",
            f"--bbox '{box1} {box2} {box3} {box4}'",
            f"--expand_num 120",
            f"--label 'car'",
            f"--image {image_path}",
            f"--maskaug_mode original",
            f"--maskaug_ratio 0.1"
        ]
        
        self._run_command(" ".join(preprocess_cmd))
        
        return self._run_inference("move_anything", prompt, output_name)


if __name__ == "__main__":
    vace = VACEWrapper()
    
    output_path = vace.create_simple_video(
        "assets/images/car.jpg",
        "A car driving on a scenic road with mountains in the background"
    )
    print(f"Generated video saved to: {output_path}")
