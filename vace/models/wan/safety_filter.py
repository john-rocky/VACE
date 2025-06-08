# -*- coding: utf-8 -*-
# Safety filter for WAN model
import re
from typing import List, Tuple, Optional

class WANSafetyFilter:
    """Safety filter to prevent generation of harmful content"""
    
    def __init__(self, language='en'):
        self.language = language
        
        # Define prohibited keywords (basic list - expand as needed)
        self.prohibited_keywords = {
            'violence': [
                'violent', 'kill', 'murder', 'torture', 'gore', 'blood',
                'weapon', 'gun', 'knife', 'assault', 'attack'
            ],
            'adult': [
                'nude', 'naked', 'nsfw', 'explicit', 'sexual',
                'pornographic', 'adult content'
            ],
            'hate': [
                'hate', 'racist', 'discrimination', 'offensive',
                'derogatory', 'slur'
            ],
            'illegal': [
                'drug', 'illegal', 'crime', 'criminal', 'theft'
            ]
        }
        
        # Safe replacement suggestions
        self.safe_replacements = {
            'violent scene': 'action scene',
            'bloody': 'dramatic',
            'weapon': 'tool',
            'attack': 'approach',
            'nude': 'artistic figure',
            'explicit': 'detailed'
        }
    
    def check_prompt(self, prompt: str) -> Tuple[bool, Optional[str], List[str]]:
        """
        Check if prompt contains prohibited content
        
        Returns:
            - is_safe (bool): Whether the prompt is safe
            - category (str): Category of violation if unsafe
            - matched_words (list): List of matched prohibited words
        """
        prompt_lower = prompt.lower()
        matched_words = []
        violated_category = None
        
        for category, keywords in self.prohibited_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    matched_words.append(keyword)
                    violated_category = category
        
        is_safe = len(matched_words) == 0
        return is_safe, violated_category, matched_words
    
    def sanitize_prompt(self, prompt: str) -> str:
        """
        Attempt to sanitize prompt by replacing problematic words
        """
        sanitized = prompt
        
        for unsafe, safe in self.safe_replacements.items():
            pattern = re.compile(re.escape(unsafe), re.IGNORECASE)
            sanitized = pattern.sub(safe, sanitized)
        
        return sanitized
    
    def get_safety_message(self, category: str, matched_words: List[str]) -> str:
        """
        Get user-friendly safety message
        """
        messages = {
            'violence': "⚠️ The prompt contains violent content. Please use more peaceful descriptions.",
            'adult': "⚠️ The prompt contains adult content. Please keep prompts appropriate.",
            'hate': "⚠️ The prompt contains potentially offensive content. Please be respectful.",
            'illegal': "⚠️ The prompt describes illegal activities. Please choose legal scenarios."
        }
        
        base_message = messages.get(category, "⚠️ The prompt contains inappropriate content.")
        word_list = ", ".join(matched_words)
        
        return f"{base_message}\nProblematic words: {word_list}"


class PromptModerator:
    """Enhanced prompt moderation with multiple safety levels"""
    
    def __init__(self, safety_level='medium'):
        """
        Initialize moderator
        
        Args:
            safety_level: 'low', 'medium', or 'high'
        """
        self.safety_level = safety_level
        self.filter = WANSafetyFilter()
        
        # Additional context-aware checks
        self.context_patterns = {
            'violence_context': [
                r'person\s+with\s+\w*weapon',
                r'holding\s+\w*gun',
                r'pointing\s+\w*at'
            ],
            'suggestive_context': [
                r'revealing\s+\w*clothing',
                r'intimate\s+\w*scene'
            ]
        }
    
    def moderate(self, prompt: str, auto_sanitize=True) -> dict:
        """
        Moderate prompt with safety checks
        
        Returns:
            dict with keys:
                - 'safe': bool
                - 'original_prompt': str
                - 'sanitized_prompt': str (if auto_sanitize)
                - 'warnings': list of warning messages
                - 'category': violation category if unsafe
        """
        # Basic keyword check
        is_safe, category, matched_words = self.filter.check_prompt(prompt)
        
        result = {
            'safe': is_safe,
            'original_prompt': prompt,
            'warnings': [],
            'category': category
        }
        
        if not is_safe:
            result['warnings'].append(self.filter.get_safety_message(category, matched_words))
            
            if auto_sanitize:
                sanitized = self.filter.sanitize_prompt(prompt)
                result['sanitized_prompt'] = sanitized
                result['warnings'].append(f"ℹ️ Prompt auto-sanitized: '{sanitized}'")
        
        # Context-aware checks for medium/high safety levels
        if self.safety_level in ['medium', 'high']:
            for context_type, patterns in self.context_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, prompt, re.IGNORECASE):
                        result['safe'] = False
                        result['warnings'].append(
                            f"⚠️ Potentially problematic context detected: {context_type}"
                        )
        
        return result


# Usage example
def apply_safety_filter(prompt: str, safety_level='medium', block_unsafe=True):
    """
    Apply safety filtering to a prompt
    
    Args:
        prompt: Input prompt
        safety_level: 'low', 'medium', or 'high'
        block_unsafe: If True, raise exception for unsafe content
    
    Returns:
        Safe prompt or raises exception
    """
    moderator = PromptModerator(safety_level)
    result = moderator.moderate(prompt)
    
    if not result['safe']:
        if block_unsafe:
            warnings = '\n'.join(result['warnings'])
            raise ValueError(f"Unsafe prompt detected:\n{warnings}")
        else:
            # Return sanitized version with warning
            print("⚠️ Safety Warning:")
            for warning in result['warnings']:
                print(warning)
            return result.get('sanitized_prompt', prompt)
    
    return prompt


if __name__ == "__main__":
    # Test examples
    test_prompts = [
        "A beautiful sunset over the ocean",
        "A person holding a weapon",
        "An artistic nude painting",
        "A peaceful garden scene"
    ]
    
    moderator = PromptModerator(safety_level='medium')
    
    for prompt in test_prompts:
        print(f"\nTesting: '{prompt}'")
        result = moderator.moderate(prompt)
        
        if result['safe']:
            print("✅ Safe prompt")
        else:
            print("❌ Unsafe prompt")
            for warning in result['warnings']:
                print(warning)