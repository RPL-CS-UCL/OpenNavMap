class FullKFSelector:
    def __init__(self):
        pass
    def select_keyframes(self, submap_database):
        """
        Main method to select keyframes from provided data.
        timestamps, descriptors, iqa_scores, info_gain: metadata dictionaries
        submap_database: list of submap dicts containing frame names
        """

        # Process each submap
        keyframes = []
        for submap in submap_database:
            keyframes += [key for key in submap['frames']]

        print(f'Selected {len(keyframes)} keyframes')
        # print(', '.join(key for key in keyframes))

        return keyframes
