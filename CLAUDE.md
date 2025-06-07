# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VACE (Video All-in-One Creation and Editing) is an AI-powered video processing framework that provides comprehensive video generation and editing capabilities. It integrates multiple state-of-the-art models including LTX-Video and WAN-Video for generation, along with various computer vision models for preprocessing.

## Commands

### Installation
```bash
# Basic installation
pip install -r requirements.txt

# Install with LTX backend support
pip install -e ".[ltx]"

# Install with WAN backend support  
pip install -e ".[wan]"

# Install annotator modules
pip install -e ".[annotator]"
```

### Running Tests
```bash
python -m pytest tests/test_annotators.py
```

### Pipeline Execution
```bash
# Basic pipeline command structure
python -m vace.vace_pipeline --base [ltx|wan] --task [TASK_NAME] --video [VIDEO_PATH] --prompt [PROMPT]

# Example: Video inpainting with LTX
python -m vace.vace_pipeline --base ltx --task inpainting --video input.mp4 --mask mask.png --prompt "your prompt"

# Example: Video generation from text
python -m vace.vace_pipeline --base ltx --task text2video --prompt "your prompt" --output_path output/
```

### Gradio Demos
```bash
# Preprocessing interface
python -m vace.gradios.vace_preprocess_demo

# LTX inference interface
python -m vace.gradios.vace_ltx_demo

# WAN inference interface  
python -m vace.gradios.vace_wan_demo
```

## Architecture

### Core Structure
- **vace/annotators/**: Computer vision preprocessing modules (depth, pose, segmentation, etc.)
- **vace/configs/**: Task configurations organized by type (image, video, composition)
- **vace/models/**: Model implementations (ltx/, wan/) with their pipelines and modules
- **vace/gradios/**: Web UI implementations using Gradio

### Key Design Patterns

1. **Configuration-Driven Tasks**: All processing tasks are defined in configuration dictionaries specifying:
   - NAME: The annotator class to use
   - INPUTS: Required input fields (video, mask, prompt, etc.)
   - OUTPUTS: Expected output fields

2. **Modular Annotators**: Each annotator inherits from common base classes and implements standardized interfaces for processing images/videos.

3. **Pipeline Architecture**: 
   - Preprocessing (vace_preproccess.py) → Inference (vace_ltx_inference.py/vace_wan_inference.py) → Output
   - Clear separation between data preparation and model execution

### Task Categories

**Image Tasks**: face, salient, inpainting, outpainting, frameref, depth, gray, pose, scribble

**Video Tasks**: depth_video, flow_video, pose_video, scribble_video, frameref_video, inpainting_video_*, outpainting_video_*, layout_video_*

**Composition Tasks**: reference_anything, animate_anything, swap_anything, expand_anything, move_anything

### Important Conventions

- Output directories are automatically created with timestamps
- Device management is handled automatically (CUDA when available)
- Configuration merging allows task-specific overrides
- Annotators cache models for efficiency across multiple calls
- Video processing maintains consistent FPS and resolution handling