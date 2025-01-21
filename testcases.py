# To be imported into main file when needing to test persona matching

test_cases = [
    {
        "input": "I want to build a warehouse automation system that uses computer vision",
        "expected_personas": ["Frida", "Alex", "Maria"],  # Known good matches
        "required_aspects": {
            "technical": ["Robotics", "Computer Vision"],
            "soft_skills": ["Safety-conscious"],
            "experience_level": ["Intermediate", "Senior"]
        }
    },
    # More test cases...
]

def evaluate_matches(collection, test_cases, config):
    results = {
        "relevance_scores": [],
        "coverage_scores": [],
        "latency_measurements": [],
        "aspect_match_scores": []
    }
    
    for test_case in test_cases:
        start_time = time.time()
        
        # Get matches from your manager agent
        matched_personas = manager_agent.find_relevant_personas(
            test_case["input"], collection
        )
        
        # Measure latency
        latency = time.time() - start_time
        results["latency_measurements"].append(latency)
        
        # Evaluate relevance (how many of the expected personas were found)
        relevance = len(set(matched_personas) & set(test_case["expected_personas"])) / \
                   len(test_case["expected_personas"])
        results["relevance_scores"].append(relevance)
        
        # Evaluate aspect coverage (technical, soft skills, experience)
        coverage = evaluate_aspect_coverage(
            matched_personas, 
            test_case["required_aspects"]
        )
        results["coverage_scores"].append(coverage)
        
    return aggregate_results(results)

configs_to_test = [
    {"M": 32, "construction_ef": 250, "search_ef": 150},
    {"M": 64, "construction_ef": 200, "search_ef": 100},
    # Add more configurations to test
]

results = {}
for config in configs_to_test:
    # Create test collection with this config
    test_collection = create_test_collection(config)
    
    # Run evaluation
    results[str(config)] = evaluate_matches(
        test_collection, 
        test_cases, 
        config
    )