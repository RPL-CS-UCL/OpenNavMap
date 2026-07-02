import numpy as np
from scipy.stats import norm

class BayesianDynamicDetector:
    """
    Bayesian framework for dynamic object probability estimation combining:
    - Semantic prior knowledge
    - Geometric error likelihoods
    
    Implements probabilistic fusion using Bayes' theorem:
    P_dynamic = (P_semantic * Likelihood_dynamic) / 
                (P_semantic * Likelihood_dynamic + (1-P_semantic) * Likelihood_static)
    
    Attributes:
        semantic_prior (dict): Prior probabilities P(dynamic) for object categories
        static_params (tuple): Gaussian parameters (mean, std) for static error distribution
        dynamic_params (tuple): Gaussian parameters (mean, std) for dynamic error distribution
    """

    def __init__(self,
                 static_params=(0.5, 0.8),
                 dynamic_params=(3.0, 1.5)):
        """
        Initialize detector with probability distributions and prior knowledge
        
        Args:
            semantic_prior: Custom prior probabilities (category: probability)
            static_params: Gaussian parameters for static objects' error distribution
            dynamic_params: Gaussian parameters for dynamic objects' error distribution
        """
        
        # Configure probability distributions
        self.static_mean, self.static_std = static_params
        self.dynamic_mean, self.dynamic_std = dynamic_params

    def calculate_per_object(self, obj):
        """
        Calculate normalized dynamic probability for a single object
        
        Process Flow:
        1. Retrieve semantic prior for object category
        2. Calculate likelihoods using static/dynamic error distributions
        3. Apply Bayesian fusion equation
        4. Return normalized probability
        
        Args:
            obj: Dictionary containing:
                - category: Semantic class (string)
                - error: Geometric error measurement (float)
                
        Returns:
            Normalized dynamic probability between 0 and 1
        """
        
        # Get category-specific prior probability
        category = obj.get('category', 'unknown')
        p_sem = obj.get('dyna_prob', 0.5)

        # Calculate probability densities for observed error
        error = obj['error']
        p_dynamic = norm.pdf(error, self.dynamic_mean, self.dynamic_std)
        p_static = norm.pdf(error, self.static_mean, self.static_std)

        # Bayesian probability fusion
        numerator = p_sem * p_dynamic
        denominator = numerator + (1 - p_sem) * p_static
        
        # Add epsilon to prevent division by zero
        return numerator / (denominator + 1e-8)

    def batch_calculate(self, objects):
        """
        Process multiple objects and return probability pairs
        
        Args:
            objects: List of object dictionaries
            
        Returns:
            Dictionary mapping object indices to:
            (P_dynamic, P_static) probability tuples
        """
        results = {}
        for obj_id, obj in enumerate(objects):
            p_dynamic = self.calculate_per_object(obj)
            results[obj_id] = (p_dynamic, 1 - p_dynamic)
        return results
    
# -----------------------------------------------------------------------------
# Demonstration and Validation
# -----------------------------------------------------------------------------
def main():
    # Test case configuration
    test_objects = [
        {'category': 'vehicle', 'error': 3.2, 'dyna_prob': 0.92},
        {'category': 'building', 'error': 0.4, 'dyna_prob': 0.1},
        {'category': 'pedestrian', 'error': 2.5, 'dyna_prob': 0.88},
        {'category': 'unknown', 'error': 1.8, 'dyna_prob': 0.5}
    ]

    # Initialize detector with default parameters
    detector = BayesianDynamicDetector(
        static_params=(0.5, 0.8),    # Static objects: low mean error
        dynamic_params=(3.0, 1.5)    # Dynamic objects: higher mean error
    )

    # Calculate probabilities for all test cases
    results = detector.batch_calculate(test_objects)

    # Display results with formatted output
    print("Dynamic vs Static Probability Analysis:")
    print("=" * 50)
    for obj_id, (p_dyn, p_stat) in results.items():
        print(f"Object {obj_id} ({test_objects[obj_id]['category']}):")
        print(f"  Dynamic probability: {p_dyn:.4f}")
        print(f"  Static probability:  {p_stat:.4f}")
        print(f"  Probability sum:     {p_dyn + p_stat:.4f} (verification)")
        print("-" * 50)

if __name__ == '__main__':
    main()