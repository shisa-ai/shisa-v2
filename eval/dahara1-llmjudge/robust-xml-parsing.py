def clean_xml(xml_string: str) -> str:
    """Clean and normalize XML string with multiple fallback mechanisms."""
    # Remove code block markers and language indicators
    clean = re.sub(r'```(?:XML)?\n?', '', xml_string)
    clean = re.sub(r'^xml\s*\n', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'<\?xml[^>]+\?>\s*', '', clean)
    
    # Fix common XML issues
    clean = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', clean)  # Fix unescaped ampersands
    clean = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', clean)   # Remove invalid XML characters
    
    # Try to extract content between <result> tags
    result_match = re.search(r'<result>.*?</result>', clean, re.DOTALL)
    if result_match:
        clean = result_match.group(0)
    else:
        # If no <result> tags found, try to extract explanation and verdict using regex
        explanation_match = re.search(r'<explanation>(.*?)</explanation>', clean, re.DOTALL)
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', clean, re.DOTALL)
        
        if explanation_match and verdict_match:
            clean = f"<result>\n  <explanation>{explanation_match.group(1)}</explanation>\n  <verdict>{verdict_match.group(1)}</verdict>\n</result>"
        elif not clean.startswith('<'):
            # Last resort: wrap everything in result tags
            clean = f'<result>\n{clean}\n</result>'
    
    # Ensure proper XML structure
    clean = clean.replace('<<', '<').replace('>>', '>')  # Fix double brackets
    clean = re.sub(r'\s+', ' ', clean)  # Normalize whitespace
    clean = clean.strip()
    
    return clean

def extract_explanation_verdict(xml_string: str) -> tuple[str, str]:
    """Extract explanation and verdict with multiple parsing strategies."""
    try:
        # First try: Standard XML parsing
        root = ET.fromstring(xml_string)
        explanation = root.find('explanation')
        verdict = root.find('verdict')
        
        if explanation is not None and verdict is not None:
            return explanation.text.strip(), verdict.text.strip()
        
        # Second try: Direct regex matching if XML parsing succeeds but elements not found
        explanation_text = ""
        verdict_text = ""
        
        explanation_match = re.search(r'<explanation>(.*?)</explanation>', xml_string, re.DOTALL)
        if explanation_match:
            explanation_text = explanation_match.group(1).strip()
        
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', xml_string, re.DOTALL)
        if verdict_match:
            verdict_text = verdict_match.group(1).strip()
        
        if explanation_text and verdict_text:
            return explanation_text, verdict_text
        
    except ET.ParseError as e:
        print(f"XML parsing failed: {str(e)}")
        # Third try: Fallback to regex with more flexible patterns
        explanation_text = ""
        verdict_text = ""
        
        # Try to find explanation
        explanation_patterns = [
            r'<explanation>(.*?)</explanation>',
            r'explanation:\s*(.*?)\s*(?:<verdict>|$)',
            r'explanation"?\s*:\s*(.*?)\s*(?:verdict|$)'
        ]
        
        for pattern in explanation_patterns:
            match = re.search(pattern, xml_string, re.DOTALL | re.IGNORECASE)
            if match:
                explanation_text = match.group(1).strip()
                break
        
        # Try to find verdict
        verdict_patterns = [
            r'<verdict>(.*?)</verdict>',
            r'verdict:\s*((?:A|B)\s+is\s+(?:much\s+)?(?:slightly\s+)?better|same)',
            r'verdict"?\s*:\s*((?:A|B)\s+is\s+(?:much\s+)?(?:slightly\s+)?better|same)'
        ]
        
        for pattern in verdict_patterns:
            match = re.search(pattern, xml_string, re.DOTALL | re.IGNORECASE)
            if match:
                verdict_text = match.group(1).strip()
                break
        
        if explanation_text or verdict_text:
            return (
                explanation_text or "Failed to parse explanation",
                verdict_text or "same"  # Default to 'same' if no verdict found
            )
    
    # Final fallback
    return "Failed to parse explanation", "same"

def validate_verdict(verdict: str) -> str:
    """Validate and normalize verdict string."""
    valid_verdicts = {
        'a is much better': 'A is much better',
        'a is better': 'A is better',
        'a is slightly better': 'A is slightly better',
        'same': 'same',
        'b is slightly better': 'B is slightly better',
        'b is better': 'B is better',
        'b is much better': 'B is much better'
    }
    
    normalized = verdict.lower().strip()
    return valid_verdicts.get(normalized, 'same')

def process_batch(llm: LLM, examples: List[Dict], sampling_params: SamplingParams) -> List[Dict]:
    # Prepare all prompts for the batch
    prompts = [
        generate_prompt(ex['prompt'], ex['response_a'], ex['response_b'])
        for ex in examples
    ]
    
    # Generate responses for all prompts in the batch
    outputs = llm.generate(prompts, sampling_params)
    
    processed_examples = []
    for i, output in enumerate(outputs):
        try:
            judge_xml = output.outputs[0].text
            clean_judge_xml = clean_xml(judge_xml)
            
            explanation, verdict = extract_explanation_verdict(clean_judge_xml)
            verdict = validate_verdict(verdict)
            
            tag = "Japanese to English" if "Japanese to English" in examples[i]['prompt'] else "English to Japanese"
            score = parse_verdict(verdict)
            
            processed_example = {
                "input_text": examples[i]['prompt'],
                "tags": [tag],
                "output_text_a": examples[i]['response_a'],
                "output_text_b": examples[i]['response_b'],
                "score": score,
                "individual_rater_scores": [],
                "custom_fields": {
                    "explanation": explanation,
                    "raw_response": judge_xml,  # Optional: keep raw response for debugging
                    "cleaned_xml": clean_judge_xml  # Optional: keep cleaned XML for debugging
                }
            }
            processed_examples.append(processed_example)
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            # Add failed example with default values
            processed_examples.append({
                "input_text": examples[i]['prompt'],
                "tags": ["Japanese to English" if "Japanese to English" in examples[i]['prompt'] else "English to Japanese"],
                "output_text_a": examples[i]['response_a'],
                "output_text_b": examples[i]['response_b'],
                "score": 0.0,  # Default to neutral score
                "individual_rater_scores": [],
                "custom_fields": {
                    "explanation": "Failed to process response",
                    "error": str(e),
                    "raw_response": judge_xml if 'judge_xml' in locals() else "No response generated"
                }
            })
    
    return processed_examples
