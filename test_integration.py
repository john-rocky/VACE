#!/usr/bin/env python3
"""
Simple test script for VACE integrated pipeline.
"""
import os
import sys
import logging
import argparse
from vace_integrated_pipeline import VACEIntegratedPipeline, generate_video_from_image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test the VACE integrated pipeline")
    parser.add_argument(
        "--image", 
        type=str, 
        default="Exterior/car9.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="A sleek car cruising smoothly on a modern highway, golden hour lighting, cinematic camera tracking shot following the vehicle, motion blur on wheels, realistic reflections on the car body, trees and scenery passing by in the background.",
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--size", 
        type=str, 
        default="832*480",
        help="Output video size (width*height)"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="pretrained_models/VACE-Wan2.1-1.3B",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="vace-1.3B",
        choices=["vace-1.3B", "vace-14B"],
        help="Model name to use"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory (auto-generated if None)"
    )
    parser.add_argument(
        "--use_convenience_function", 
        action="store_true",
        help="Use the convenience function instead of the class"
    )
    return parser.parse_args()

def test_pipeline(args):
    """Test the integrated pipeline with provided arguments."""
    
    if not os.path.exists(args.image):
        logging.error(f"Test image {args.image} not found. Please provide a valid image path.")
        return False
    
    try:
        if args.use_convenience_function:
            logging.info("Using convenience function")
            video_path = generate_video_from_image(
                args.image, 
                args.prompt, 
                args.size, 
                args.checkpoint_dir,
                args.model_name
            )
        else:
            logging.info("Using pipeline class")
            pipeline = VACEIntegratedPipeline(
                checkpoint_dir=args.checkpoint_dir,
                model_name=args.model_name
            )
            
            video_path = pipeline.generate_video(
                args.image, 
                args.prompt, 
                args.size,
                args.output_dir
            )
        
        logging.info(f"Successfully generated video: {video_path}")
        return os.path.exists(video_path)
        
    except Exception as e:
        logging.error(f"Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    args = parse_args()
    success = test_pipeline(args)
    sys.exit(0 if success else 1)
