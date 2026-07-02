import copy
import numpy as np
import argparse
from bayesian_dynamic_detector import BayesianDynamicDetector

def detect_change(input_mask_path):
    """
    Process instance masks with Bayesian dynamic probability estimation
    
    Workflow:
    1. Load initial instance masks
    2. Initialize Bayesian detector
    3. Update dynamic probabilities
    4. Generate combined scene mask
    5. Save processed data
    
    File Structure:
    - instance_masks: List of dicts with object properties
    - scene_mask: 2D array with object IDs at pixel locations
    """
    
    # Load initial masks with safety checks
    masks = np.load(input_mask_path, allow_pickle=True)
    instance_masks = masks.item().get('instance_masks', [])

    # Initialize Bayesian detector with domain-specific priors
    detector = BayesianDynamicDetector(
        static_params=(0.3, 0.5),
        dynamic_params=(2.5, 1.2)
    )

    # Update probabilities for each object
    for i in range(10):
        update_instance_masks = []
        for obj in instance_masks:
            obj_info = copy.deepcopy(obj)
            obj_info['category'] = obj['category']
            obj_info['error'] = 1.0

            dyna_prob = detector.calculate_per_object(obj_info)
            obj_info['dyna_prob'] = dyna_prob
            update_instance_masks.append(obj_info)

        P_ori = [obj['dyna_prob'] for obj in instance_masks]
        P_upate = [obj['dyna_prob'] for obj in update_instance_masks]
        print(f"{i}th")
        print(P_ori)
        print(P_upate)
        print()

        instance_masks = update_instance_masks

    # Generate combined scene mask
    static_mask = generate_static_mask(update_instance_masks)
   
    return update_instance_masks, static_mask

def generate_static_mask(instance_masks):
    """Create 2D label matrix from processed masks"""
    if not instance_masks:
        return np.zeros((0, 0), dtype=bool)
    
    h, w = instance_masks[0]['segmentation'].shape
    static_mask = np.zeros((h, w), dtype=bool)
    static_mask.fill(True)

    for idx, mask in enumerate(instance_masks, 1):
        dyn_mask = mask['segmentation'] & (mask['dyna_prob'] > 0.7)
        static_mask[dyn_mask] = False
        
    return static_mask

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Dynamic Object Mask Processor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-i', '--input_mask',
                       default='init_masks.npy',
                       help='Path to input init masks NPY file')
                       
    parser.add_argument('-s', '--output_mask',
                       default='update_masks.npy',
                       help='Output path for update scene mask NPY file')
                       
    args = parser.parse_args()
    
    try:
        update_instance_masks, update_static_mask = detect_change(args.input_mask)
        np.save(args.output_mask, {
            'instance_masks': update_instance_masks,
            'static_mask': update_static_mask
        })
        print(f"Successfully processed")
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
