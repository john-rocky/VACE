# VACE Python Wrapper

A Python class wrapper for the VACE repository that provides direct function calls for video generation without subprocess.

## Usage

```python
from vace_wrapper import VACEWrapper

# Initialize the wrapper
wrapper = VACEWrapper(
    model_name="vace-1.3B",
    ckpt_dir="models/Wan2.1-VACE-1.3B/"
)

# Generate simple I2V video
video_path = wrapper.create_simple_video(
    reference_image="path/to/image.jpg",
    prompt="A beautiful scene with motion",
    frame_num=81
)

# Generate video with reference using swap_anything
video_path = wrapper.create_referenced_video(
    reference_image="path/to/image.jpg", 
    reference_video="path/to/video.mp4",
    prompt="Transform the scene with new elements",
    mode="masktrack,salient"
)
```

## Methods

### `create_simple_video(reference_image, prompt, **kwargs)`
Generate video from reference image and prompt using image_reference task.

### `create_referenced_video(reference_image, reference_video, prompt, **kwargs)`
Generate video using swap_anything with reference image and video.

## Parameters

- `reference_image`: Path to reference image
- `reference_video`: Path to reference video (for referenced method)
- `prompt`: Text description for video generation
- `output_dir`: Output directory (optional)
- `frame_num`: Number of frames (default: 81)
- `size`: Video resolution (default: "480p")
- Additional inference parameters can be passed as kwargs

## Requirements

- CUDA-capable GPU
- All VACE model dependencies installed
- Model checkpoints downloaded
