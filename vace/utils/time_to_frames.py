# -*- coding: utf-8 -*-
# Utility to convert time duration to frame numbers

def time_to_frames(seconds, fps=16, wan_compatible=True):
    """
    Convert time in seconds to frame count
    
    Args:
        seconds: Duration in seconds
        fps: Frames per second (default 16 for WAN)
        wan_compatible: Ensure frame count is 4n+1 for WAN
    
    Returns:
        Frame count
    """
    frames = int(seconds * fps)
    
    if wan_compatible:
        # WAN requires 4n+1 format
        # Find nearest valid frame count
        remainder = (frames - 1) % 4
        if remainder != 0:
            # Round to nearest 4n+1
            frames = frames + (4 - remainder)
    
    return frames


def frames_to_time(frames, fps=16):
    """
    Convert frame count to time in seconds
    
    Args:
        frames: Number of frames
        fps: Frames per second
    
    Returns:
        Duration in seconds
    """
    return frames / fps


def get_wan_valid_frames(min_frames=1, max_frames=200):
    """
    Get list of valid frame counts for WAN (4n+1 format)
    """
    valid_frames = []
    n = 0
    while True:
        frame_count = 4 * n + 1
        if frame_count > max_frames:
            break
        if frame_count >= min_frames:
            valid_frames.append(frame_count)
        n += 1
    return valid_frames


def suggest_frame_count(target_seconds, fps=16):
    """
    Suggest the best frame count for target duration
    
    Args:
        target_seconds: Desired duration in seconds
        fps: Frames per second
    
    Returns:
        Dictionary with suggestions
    """
    target_frames = target_seconds * fps
    valid_frames = get_wan_valid_frames(1, int(target_frames * 1.5))
    
    # Find closest valid frame count
    closest = min(valid_frames, key=lambda x: abs(x - target_frames))
    
    # Also find shorter and longer options
    shorter = [f for f in valid_frames if f < closest]
    longer = [f for f in valid_frames if f > closest]
    
    suggestions = {
        'target_seconds': target_seconds,
        'target_frames': int(target_frames),
        'recommended': {
            'frames': closest,
            'seconds': frames_to_time(closest, fps),
            'command': f'--frame_num {closest}'
        }
    }
    
    if shorter:
        suggestions['shorter'] = {
            'frames': shorter[-1],
            'seconds': frames_to_time(shorter[-1], fps),
            'command': f'--frame_num {shorter[-1]}'
        }
    
    if longer:
        suggestions['longer'] = {
            'frames': longer[0],
            'seconds': frames_to_time(longer[0], fps),
            'command': f'--frame_num {longer[0]}'
        }
    
    return suggestions


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python time_to_frames.py <seconds> [fps]")
        print("\nExamples:")
        print("  python time_to_frames.py 5      # 5 seconds at 16fps")
        print("  python time_to_frames.py 3 24   # 3 seconds at 24fps")
        print("\nValid WAN frame counts (4n+1):")
        print(get_wan_valid_frames(1, 100))
        sys.exit(1)
    
    seconds = float(sys.argv[1])
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    
    suggestions = suggest_frame_count(seconds, fps)
    
    print(f"\n🎬 Time to Frame Conversion")
    print(f"{'='*40}")
    print(f"Target: {seconds} seconds at {fps}fps")
    print(f"Raw frames: {suggestions['target_frames']}")
    
    print(f"\n✅ Recommended (closest):")
    rec = suggestions['recommended']
    print(f"  {rec['frames']} frames = {rec['seconds']:.2f} seconds")
    print(f"  Command: {rec['command']}")
    
    if 'shorter' in suggestions:
        short = suggestions['shorter']
        print(f"\n⏪ Shorter option:")
        print(f"  {short['frames']} frames = {short['seconds']:.2f} seconds")
        print(f"  Command: {short['command']}")
    
    if 'longer' in suggestions:
        long = suggestions['longer']
        print(f"\n⏩ Longer option:")
        print(f"  {long['frames']} frames = {long['seconds']:.2f} seconds")
        print(f"  Command: {long['command']}")