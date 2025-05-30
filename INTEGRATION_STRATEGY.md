# VACE Pipeline Integration Strategy

## Overview
This document outlines the implementation strategy for integrating VACE preprocessing and inference into a single Python class. The implementation combines the functionality of `vace_preproccess.py` and `vace_wan_inference.py` into a unified pipeline that takes an image and prompt as input and returns a generated video path.

## Architecture

### Key Components
1. **SubjectAnnotator**: Handles image preprocessing with salient mode detection
2. **WanVace**: Performs text-to-video generation using diffusion models  
3. **VaceVideoProcessor**: Handles data preparation and format conversion
4. **VACEIntegratedPipeline**: Main integration class

### Data Flow
```
Input Image → SubjectAnnotator → Processed Image + Mask → WanVace → Output Video
```

## Implementation Details

### Model Loading Strategy
- Models are loaded once in `__init__` method for reuse across multiple executions
- WanVace model initialization includes text encoder, VAE, and diffusion model components
- SubjectAnnotator loads U2NET salient detection model and other required annotators

### Configuration Management
- Reuses existing `VACE_CONFIGS` dictionary for preprocessing task configuration
- WAN model configuration loaded from `WAN_CONFIGS` dictionary
- Maintains compatibility with existing model checkpoint format
- Supports both model variants: vace-1.3B and vace-14B

### File Management
- Creates timestamped output directories matching original script behavior
- Generates intermediate files (src_ref_image, src_mask) for debugging/inspection
- Returns final video path for easy consumption

### Error Handling
- Comprehensive exception handling with informative error messages
- Graceful degradation when optional parameters are missing
- Validation of input file existence and format

## Implementation Challenges

### Preprocessing Integration
The preprocessing step uses the `SubjectAnnotator` class with the "salient" mode to process the input image. This generates both a processed reference image and a mask, which are saved to disk and then used as input for the inference step.

### Model Initialization
The WanVace model requires specific configuration parameters and checkpoint paths. The implementation loads these from the existing configuration dictionaries in the VACE codebase to ensure compatibility.

### Video Generation
The inference step uses the WanVace model to generate a video based on the preprocessed image, mask, and text prompt. The implementation handles the data preparation, model inference, and video saving steps.

## Usage Examples

### Basic Usage
```python
from vace_integrated_pipeline import VACEIntegratedPipeline

pipeline = VACEIntegratedPipeline()
video_path = pipeline.generate_video(
    "path/to/image.jpg", 
    "A sleek car cruising on a highway"
)
```

### Advanced Usage with Custom Configuration
```python
pipeline = VACEIntegratedPipeline(
    checkpoint_dir="custom/model/path",
    model_name="vace-14B"
)
video_path = pipeline.generate_video(
    "path/to/image.jpg",
    "Custom prompt", 
    size="1280*720",
    output_dir="custom/output/dir"
)
```

### Convenience Function
```python
from vace_integrated_pipeline import generate_video_from_image

video_path = generate_video_from_image(
    "path/to/image.jpg",
    "A sleek car cruising on a highway",
    size=(832, 480)
)
```

## Performance Considerations
- Model loading is done once during initialization to avoid repeated loading
- The pipeline can be reused for multiple executions
- GPU memory usage is optimized by offloading models when not in use

## Future Improvements
- Add support for additional preprocessing modes (mask, bbox, etc.)
- Implement batch processing for multiple images
- Add progress reporting for long-running operations
- Optimize memory usage for large models
