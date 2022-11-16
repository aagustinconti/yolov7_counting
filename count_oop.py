

class Count():
    def __init__(self, ds_output, roi, names) -> None:

        self.count_out_classes = {}
        counted = []

        for detection in ds_output:

            # Get variables
            ds_cpoint = detection[0]
            ds_id = detection[1]
            ds_class = detection[2]

            # To check if the ds_cpoint is into the roi
            is_into_roi = (roi[0] < ds_cpoint[0] < roi[2]) and (
                roi[1] < ds_cpoint[1] < roi[3])

            # If is into the roi
            if is_into_roi:

                # fill the empty vector
                if len(counted) == 0:
                    counted.append([ds_id, ds_class])

                # get the classes detected
                    self.count_out_classes = dict.fromkeys(
                        [elem[1] for elem in counted], 0)

                    # count per class
                    for elem in counted:
                        self.count_out_classes[elem[1]] += 1

                else:
                    # if the id is not in the list
                    if (ds_id not in [elem[0] for elem in counted]):
                        # count object
                        counted.append([ds_id, ds_class])

                        # get the classes detected
                        self.count_out_classes = dict.fromkeys(
                            [elem[1] for elem in counted], 0)

                        # count per class
                        for elem in counted:
                            self.count_out_classes[elem[1]] += 1

        self.counter_text = [[key, names[key], self.count_out_classes[key]]
                             for key in self.count_out_classes.keys()]

    def __str__(self) -> str:
        output_text_counting = f"""
        COUNTING:\n
        Classes Detected: {self.count_out_classes}
        Counter output: {self.counter_text}
    
        """
        return output_text_counting
