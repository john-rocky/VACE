import os
import sys
from vace_wrapper import VACEWrapper

class MockVACEWrapper(VACEWrapper):
    """A mock version of VACEWrapper that bypasses model verification and file system operations."""
    
    def _verify_vace_installation(self):
        """Override to bypass model verification."""
        vace_preprocess_path = os.path.join(self.vace_dir, "vace/vace_preproccess.py")
        vace_inference_path = os.path.join(self.vace_dir, "vace/vace_wan_inference.py")
        
        if not os.path.exists(vace_preprocess_path) or not os.path.exists(vace_inference_path):
            raise FileNotFoundError(f"VACE scripts not found in {self.vace_dir}. Please provide the correct VACE directory.")
        
        print(f"Skipping model verification for testing purposes")
    
    def _get_latest_processed_dir(self, task):
        """Override to bypass directory checking."""
        print(f"Simulating processed directory for task: {task}")
        return f"/mock/processed/{task}/timestamp"
    
    def _get_processed_files(self, task):
        """Override to return mock file paths."""
        print(f"Simulating processed files for task: {task}")
        return {
            "src_video": f"/mock/processed/{task}/src_video-{task}.mp4",
            "src_mask": f"/mock/processed/{task}/src_mask-{task}.mp4",
            "src_ref_images": f"/mock/processed/{task}/src_ref_image_0-{task}.png"
        }
    
    def _run_command(self, command):
        """Override to simulate command execution without actually running it."""
        print(f"Would execute: {command}")
        
        if "vace_preproccess.py" in command:
            print("Simulating successful preprocessing")
            return "Mock preprocessing output"
        
        if "vace_wan_inference.py" in command:
            if "--save_file" not in command and "--save_dir" not in command:
                raise ValueError("Command does not use --save_file or --save_dir parameter")
            if "--output" in command:
                raise ValueError("Command incorrectly uses --output parameter")
            
            output_path = None
            parts = command.split()
            for i, part in enumerate(parts):
                if part == "--save_file" and i + 1 < len(parts):
                    output_path = parts[i + 1]
                    break
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    f.write("Mock video file")
                print(f"Created mock output file at {output_path}")
        
        return "Mock command output"

def main():
    """Test the VACEWrapper class functionality."""
    print("Testing VACEWrapper class...")
    
    vace = MockVACEWrapper()
    
    image_path = os.path.join("assets", "images", "car.jpg")
    prompt = "A car driving on a scenic road with mountains in the background"
    
    print(f"Using image: {image_path}")
    print(f"Using prompt: {prompt}")
    
    try:
        output_path = vace.create_simple_video(
            image_path,
            prompt,
            output_name="test_simple_video.mp4"
        )
        print(f"Success! Generated video saved to: {output_path}")
        print(f"File exists: {os.path.exists(output_path)}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
        
        print("\nTesting create_animation_video method...")
        output_path = vace.create_animation_video(
            image_path,
            prompt,
            output_name="test_animation_video.mp4"
        )
        print(f"Success! Generated video saved to: {output_path}")
        
        print("\nTesting create_run_video method...")
        output_path = vace.create_run_video(
            image_path,
            prompt,
            output_name="test_run_video.mp4"
        )
        print(f"Success! Generated video saved to: {output_path}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
