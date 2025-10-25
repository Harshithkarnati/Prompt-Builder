#!/usr/bin/env python3
"""
Enhanced T5 Fine-tuning Dataset Builder

This improved script processes a JSONL file containing vague prompts and template matches
to create a high-quality T5 fine-tuning dataset. 

The key improvements include:
- Better template selection and processing
- More intelligent placeholder filling
- Context-aware prompt generation
- Cleaner output formatting

Author: AI Assistant
Date: October 2025
"""

import json
import re
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import random


class EnhancedT5DatasetBuilder:
    """Enhanced T5 dataset builder with better template processing."""
    
    def __init__(self, input_file: str, output_file: str):
        """Initialize the enhanced dataset builder."""
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def analyze_vague_prompt(self, vague_prompt: str) -> Dict[str, Any]:
        """
        Analyze the vague prompt to extract key information.
        
        Args:
            vague_prompt: User's original request
            
        Returns:
            Dictionary with analysis results
        """
        prompt_lower = vague_prompt.lower()
        
        analysis = {
            'intent': 'general',
            'content_type': 'text',
            'domain': 'general',
            'keywords': [],
            'action_words': []
        }
        
        # Detect intent
        if any(word in prompt_lower for word in ['write', 'create', 'generate', 'make']):
            analysis['intent'] = 'creation'
        elif any(word in prompt_lower for word in ['help', 'assist', 'guide', 'teach']):
            analysis['intent'] = 'assistance'
        elif any(word in prompt_lower for word in ['improve', 'optimize', 'better', 'enhance']):
            analysis['intent'] = 'improvement'
        
        # Detect content type
        content_types = {
            'social media': ['social', 'media', 'post', 'instagram', 'twitter', 'facebook'],
            'presentation': ['presentation', 'present', 'speaking', 'slides'],
            'email': ['email', 'newsletter', 'message'],
            'business': ['business', 'sales', 'marketing', 'strategy'],
            'technical': ['technical', 'documentation', 'code', 'system'],
            'creative': ['creative', 'story', 'content', 'engaging']
        }
        
        for content_type, keywords in content_types.items():
            if any(keyword in prompt_lower for keyword in keywords):
                analysis['content_type'] = content_type
                break
        
        # Extract keywords (simple approach)
        words = re.findall(r'\b\w+\b', prompt_lower)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'me', 'my', 'you', 'your', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'need', 'want', 'get', 'help', 'make', 'something', 'about'}
        analysis['keywords'] = [word for word in words if word not in stop_words and len(word) > 2]
        
        return analysis
    
    def select_best_template(self, template_matches: List[Dict[str, Any]], vague_prompt: str) -> Dict[str, Any]:
        """
        Select the most appropriate template based on relevance score and content analysis.
        """
        if not template_matches:
            raise ValueError("No template matches provided")
        
        # Analyze the vague prompt
        prompt_analysis = self.analyze_vague_prompt(vague_prompt)
        
        # Score templates based on content alignment
        scored_templates = []
        
        for template in template_matches:
            template_name = template.get('template_name', '').lower()
            template_content = template.get('template_content', '').lower()
            base_score = template.get('relevance_score', 0)
            
            # Bonus points for content type alignment
            content_bonus = 0
            if prompt_analysis['content_type'] == 'social media' and 'social' in template_name:
                content_bonus = 0.1
            elif prompt_analysis['content_type'] == 'presentation' and any(word in template_name for word in ['presentation', 'speaking']):
                content_bonus = 0.1
            elif prompt_analysis['content_type'] == 'email' and 'email' in template_name:
                content_bonus = 0.1
            elif prompt_analysis['content_type'] == 'business' and any(word in template_name for word in ['business', 'sales', 'marketing']):
                content_bonus = 0.1
            
            final_score = base_score + content_bonus
            scored_templates.append((template, final_score))
        
        # Return the highest scoring template
        best_template, _ = max(scored_templates, key=lambda x: x[1])
        return best_template
    
    def generate_contextual_content(self, vague_prompt: str, template: Dict[str, Any], user_metadata: Dict[str, Any]) -> str:
        """
        Generate a contextual, optimized prompt based on the vague prompt and template.
        
        This is the core function that creates the final AI-ready prompt.
        """
        template_name = template.get('template_name', '')
        template_content = template.get('template_content', '')
        
        # Analyze the vague prompt
        prompt_analysis = self.analyze_vague_prompt(vague_prompt)
        
        # Extract user context
        user_type = user_metadata.get('user_type', 'General User')
        style_pref = user_metadata.get('style_preference', '')
        depth_pref = user_metadata.get('depth_preference', 'Medium')
        speaking_ability = user_metadata.get('speaking_ability', 'Average')
        
        # Generate a contextual prompt based on template type and user request
        if 'Social Media' in template_name:
            return self._generate_social_media_prompt(vague_prompt, template_content, user_metadata, prompt_analysis)
        elif 'Email' in template_name:
            return self._generate_email_prompt(vague_prompt, template_content, user_metadata, prompt_analysis)
        elif any(word in template_name.lower() for word in ['presentation', 'speaking']):
            return self._generate_presentation_prompt(vague_prompt, template_content, user_metadata, prompt_analysis)
        elif any(word in template_name.lower() for word in ['business', 'sales', 'marketing']):
            return self._generate_business_prompt(vague_prompt, template_content, user_metadata, prompt_analysis)
        elif 'Creative' in template_name or 'Story' in template_name:
            return self._generate_creative_prompt(vague_prompt, template_content, user_metadata, prompt_analysis)
        else:
            return self._generate_general_prompt(vague_prompt, template_content, user_metadata, prompt_analysis)
    
    def _generate_social_media_prompt(self, vague_prompt: str, template_content: str, user_metadata: Dict[str, Any], prompt_analysis: Dict[str, Any]) -> str:
        """Generate optimized social media content prompt."""
        user_type = user_metadata.get('user_type', 'General User')
        style_pref = user_metadata.get('style_preference', '')
        
        # Determine platform and style
        platform = 'multiple platforms'
        if 'instagram' in vague_prompt.lower():
            platform = 'Instagram'
        elif 'twitter' in vague_prompt.lower() or 'x' in vague_prompt.lower():
            platform = 'Twitter/X'
        elif 'linkedin' in vague_prompt.lower():
            platform = 'LinkedIn'
        
        # Determine content focus
        content_focus = 'engaging content'
        if prompt_analysis['keywords']:
            main_topics = prompt_analysis['keywords'][:3]
            content_focus = f"content about {', '.join(main_topics)}"
        
        # Build optimized prompt
        prompt = f"""Create compelling social media content for {platform} that resonates with your audience.

**Content Goal:** {content_focus}
**Target Audience:** {user_type.lower()}s and similar demographics
**Tone:** {'Professional and polished' if style_pref == 'Professional' else 'Engaging and authentic' if style_pref == 'Creative' else 'Clear and accessible'}

**Content Guidelines:**
- Write content that encourages engagement and interaction
- Include relevant hashtags to increase discoverability
- Use a hook in the first line to grab attention
- Keep the message focused and valuable to your audience
- Include a clear call-to-action when appropriate

**Format Requirements:**
- Start with an attention-grabbing opening
- Structure content for easy readability (short paragraphs, bullet points if needed)
- End with an engaging question or call-to-action
- Suggest 3-5 relevant hashtags

Please create the social media content based on this request: "{vague_prompt}"
"""
        
        return prompt.strip()
    
    def _generate_email_prompt(self, vague_prompt: str, template_content: str, user_metadata: Dict[str, Any], prompt_analysis: Dict[str, Any]) -> str:
        """Generate optimized email content prompt."""
        user_type = user_metadata.get('user_type', 'Professional')
        
        email_type = 'professional email'
        if 'newsletter' in vague_prompt.lower():
            email_type = 'newsletter'
        elif 'marketing' in vague_prompt.lower():
            email_type = 'marketing email'
        
        prompt = f"""Create an effective {email_type} that achieves your communication goals.

**Email Purpose:** Based on your request: "{vague_prompt}"
**Sender Profile:** {user_type}
**Communication Style:** Professional yet personable

**Email Structure:**
1. **Subject Line:** Clear, specific, and compelling
2. **Opening:** Warm greeting and context setting
3. **Main Content:** 
   - Clear value proposition or main message
   - Organized information (bullets or short paragraphs)
   - Relevant details that serve the reader
4. **Call-to-Action:** Specific next steps if needed
5. **Professional Closing:** Appropriate sign-off

**Writing Guidelines:**
- Keep paragraphs short and scannable
- Use active voice and clear language
- Focus on the recipient's needs and interests
- Maintain a professional but human tone
- Include specific details relevant to your request

Please write the email content for: "{vague_prompt}"
"""
        
        return prompt.strip()
    
    def _generate_presentation_prompt(self, vague_prompt: str, template_content: str, user_metadata: Dict[str, Any], prompt_analysis: Dict[str, Any]) -> str:
        """Generate optimized presentation content prompt."""
        user_type = user_metadata.get('user_type', 'Professional')
        speaking_ability = user_metadata.get('speaking_ability', 'Average')
        depth_pref = user_metadata.get('depth_preference', 'Medium')
        
        audience_level = 'general audience'
        if 'executive' in vague_prompt.lower():
            audience_level = 'executive audience'
        elif 'technical' in vague_prompt.lower():
            audience_level = 'technical audience'
        
        detail_level = 'moderately detailed'
        if depth_pref == 'Comprehensive':
            detail_level = 'comprehensive and detailed'
        elif depth_pref == 'Basic':
            detail_level = 'concise and focused'
        
        prompt = f"""Create an impactful presentation that effectively communicates your message to your audience.

**Presentation Goal:** {vague_prompt}
**Speaker Profile:** {user_type} with {speaking_ability.lower()} speaking experience
**Target Audience:** {audience_level}
**Content Depth:** {detail_level}

**Presentation Structure:**
1. **Opening Hook:** Compelling start that grabs attention
2. **Clear Objective:** What the audience will learn or gain
3. **Main Content:** 
   - 3-5 key points maximum
   - Supporting evidence or examples for each point
   - Logical flow between ideas
4. **Interactive Elements:** Questions or engagement opportunities
5. **Strong Conclusion:** Summary and memorable takeaway
6. **Call-to-Action:** Next steps for the audience

**Delivery Considerations:**
- Use conversational, confident language
- Include transitions between sections
- Build in natural pauses for emphasis
- Anticipate and address potential questions
- Keep technical jargon appropriate for audience level

**Visual Support Suggestions:**
- Key statistics or data points
- Relevant images or diagrams
- Bullet points for complex information

Please create the presentation content addressing: "{vague_prompt}"
"""
        
        return prompt.strip()
    
    def _generate_business_prompt(self, vague_prompt: str, template_content: str, user_metadata: Dict[str, Any], prompt_analysis: Dict[str, Any]) -> str:
        """Generate optimized business content prompt."""
        user_type = user_metadata.get('user_type', 'Business Professional')
        style_pref = user_metadata.get('style_preference', 'Professional')
        
        business_context = 'general business'
        if any(word in vague_prompt.lower() for word in ['sales', 'sell', 'revenue']):
            business_context = 'sales and revenue'
        elif any(word in vague_prompt.lower() for word in ['marketing', 'promote', 'brand']):
            business_context = 'marketing and branding'
        elif any(word in vague_prompt.lower() for word in ['strategy', 'plan', 'growth']):
            business_context = 'strategic planning'
        
        prompt = f"""Develop professional business content that drives results and achieves your objectives.

**Business Context:** {business_context}
**Request:** {vague_prompt}
**Professional Level:** {user_type}
**Communication Style:** {style_pref if style_pref else 'Professional and results-oriented'}

**Content Framework:**
1. **Executive Summary:** Clear overview of the main points
2. **Current Situation:** Context and background
3. **Objectives:** Specific, measurable goals
4. **Strategy/Approach:** 
   - Methodical action plan
   - Key tactics and implementations
   - Timeline considerations
5. **Expected Outcomes:** Measurable results and benefits
6. **Next Steps:** Concrete actions to take

**Business Writing Principles:**
- Lead with the bottom line and key insights
- Use data and evidence to support recommendations
- Write with clarity and professional authority
- Focus on ROI and business impact
- Include specific, actionable recommendations

**Deliverable Format:**
- Professional structure with clear headers
- Executive-level language appropriate for decision makers
- Bulleted action items where appropriate
- Quantifiable metrics when possible

Create comprehensive business content for: "{vague_prompt}"
"""
        
        return prompt.strip()
    
    def _generate_creative_prompt(self, vague_prompt: str, template_content: str, user_metadata: Dict[str, Any], prompt_analysis: Dict[str, Any]) -> str:
        """Generate optimized creative content prompt."""
        style_pref = user_metadata.get('style_preference', 'Creative')
        user_type = user_metadata.get('user_type', 'Creative Professional')
        
        creative_type = 'creative content'
        if 'story' in vague_prompt.lower():
            creative_type = 'story or narrative'
        elif 'content' in vague_prompt.lower():
            creative_type = 'engaging content'
        elif 'writing' in vague_prompt.lower():
            creative_type = 'creative writing'
        
        prompt = f"""Create compelling {creative_type} that captures attention and engages your audience.

**Creative Brief:** {vague_prompt}
**Creator Profile:** {user_type}
**Style Direction:** {style_pref if style_pref else 'Creative and engaging'}

**Creative Elements:**
1. **Hook/Opening:** Immediate attention-grabber
2. **Voice and Tone:** 
   - Authentic and relatable
   - Appropriate for your audience
   - Consistent throughout
3. **Narrative Structure:**
   - Clear beginning, development, and conclusion
   - Logical flow with engaging transitions
   - Emotional resonance with readers
4. **Unique Value:** What makes this content distinctive
5. **Call-to-Action:** How readers should respond or engage

**Creative Guidelines:**
- Use vivid, descriptive language that paints pictures
- Include sensory details when appropriate
- Vary sentence structure for rhythm and flow
- Show rather than tell when possible
- Create emotional connection with your audience
- End with a memorable impression

**Content Considerations:**
- Keep your target audience in mind
- Balance creativity with clarity
- Use metaphors or analogies to explain complex ideas
- Include relevant examples or anecdotes
- Ensure the content serves a clear purpose

Develop creative content based on: "{vague_prompt}"
"""
        
        return prompt.strip()
    
    def _generate_general_prompt(self, vague_prompt: str, template_content: str, user_metadata: Dict[str, Any], prompt_analysis: Dict[str, Any]) -> str:
        """Generate optimized general-purpose prompt."""
        user_type = user_metadata.get('user_type', 'User')
        speaking_ability = user_metadata.get('speaking_ability', 'Average')
        depth_pref = user_metadata.get('depth_preference', 'Medium')
        
        complexity_level = 'intermediate'
        if speaking_ability == 'Basic' or user_type in ['Casual User', 'Beginner']:
            complexity_level = 'beginner-friendly'
        elif speaking_ability == 'Expert' or user_type in ['Expert', 'Professional']:
            complexity_level = 'advanced'
        
        detail_level = 'moderate detail'
        if depth_pref == 'Comprehensive':
            detail_level = 'comprehensive depth'
        elif depth_pref == 'Basic':
            detail_level = 'concise and focused'
        
        prompt = f"""Provide helpful, accurate, and well-structured assistance for your request.

**Request:** {vague_prompt}
**User Profile:** {user_type}
**Complexity Level:** {complexity_level}
**Detail Level:** {detail_level}

**Response Framework:**
1. **Understanding:** Clear interpretation of your request
2. **Context:** Relevant background information
3. **Main Content:**
   - Structured, logical presentation of information
   - Step-by-step guidance where appropriate
   - Examples and practical applications
4. **Key Takeaways:** Most important points to remember
5. **Next Steps:** Suggested actions or further resources

**Communication Guidelines:**
- Use clear, accessible language appropriate for your experience level
- Organize information in logical, easy-to-follow sections
- Provide specific, actionable advice
- Include relevant examples when helpful
- Address potential questions or concerns
- Focus on practical, usable information

**Quality Standards:**
- Accurate and up-to-date information
- Well-organized and easy to scan
- Appropriate depth for your needs
- Professional yet approachable tone
- Actionable recommendations

Please provide comprehensive assistance for: "{vague_prompt}"
"""
        
        return prompt.strip()
    
    def create_input_text(self, vague_prompt: str, user_metadata: Dict[str, Any]) -> str:
        """Create clean input text for T5 training."""
        input_parts = [f"Optimize this request: {vague_prompt}"]
        
        # Add user context
        if user_metadata.get('user_type'):
            input_parts.append(f"User: {user_metadata['user_type']}")
        
        if user_metadata.get('style_preference'):
            input_parts.append(f"Style: {user_metadata['style_preference']}")
        
        if user_metadata.get('depth_preference'):
            input_parts.append(f"Depth: {user_metadata['depth_preference']}")
        
        if user_metadata.get('speaking_ability'):
            input_parts.append(f"Level: {user_metadata['speaking_ability']}")
        
        return " | ".join(input_parts)
    
    def process_entry(self, entry: Dict[str, Any]) -> Dict[str, str]:
        """Process a single JSONL entry into optimized T5 training format."""
        vague_prompt = entry.get('vague_prompt', '')
        template_matches = entry.get('template_matches', [])
        
        user_metadata = {
            'user_type': entry.get('user_type'),
            'style_preference': entry.get('style_preference'),
            'depth_preference': entry.get('depth_preference'),
            'speaking_ability': entry.get('speaking_ability')
        }
        
        # Select best template
        best_template = self.select_best_template(template_matches, vague_prompt)
        
        # Generate optimized prompt
        target_text = self.generate_contextual_content(vague_prompt, best_template, user_metadata)
        
        # Create input text
        input_text = self.create_input_text(vague_prompt, user_metadata)
        
        return {
            'input_text': input_text,
            'target_text': target_text
        }
    
    def build_dataset(self) -> None:
        """Build the complete enhanced T5 dataset."""
        print(f"Building enhanced T5 dataset...")
        print(f"Input: {self.input_file}")
        print(f"Output: {self.output_file}")
        
        processed_count = 0
        error_count = 0
        
        with open(self.input_file, 'r', encoding='utf-8') as infile, \
             open(self.output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    entry = json.loads(line.strip())
                    result = self.process_entry(entry)
                    
                    json.dump(result, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} entries...")
                
                except Exception as e:
                    error_count += 1
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue
        
        print(f"\nEnhanced dataset building complete!")
        print(f"Successfully processed: {processed_count} entries")
        print(f"Errors encountered: {error_count} entries")
        print(f"Output saved to: {self.output_file}")


def main():
    """Main function to run the enhanced dataset builder."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    input_file = project_root / "data" / "retrieval" / "pure_python_results.jsonl"
    output_file = project_root / "data" / "training" / "t5_dataset.jsonl"
    
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    
    try:
        builder = EnhancedT5DatasetBuilder(str(input_file), str(output_file))
        builder.build_dataset()
        
        # Show sample outputs
        print("\n" + "="*60)
        print("SAMPLE ENHANCED OUTPUTS:")
        print("="*60)
        
        with open(output_file, 'r', encoding='utf-8') as f:
            # Show first few samples
            for i in range(3):
                try:
                    sample = json.loads(f.readline())
                    print(f"\n--- SAMPLE {i+1} ---")
                    print(f"INPUT: {sample['input_text']}")
                    print(f"TARGET: {sample['target_text'][:300]}..." if len(sample['target_text']) > 300 else f"TARGET: {sample['target_text']}")
                except:
                    break
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()