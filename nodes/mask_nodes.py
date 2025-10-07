import numpy as np
import torch

class ReplaceEmptyMasksWithLastMasks:

    RETURN_TYPES = ("MASK", )
    FUNCTION = "replaceemptywithlastmasks"
    CATEGORY = "badnodes/image"
    DESCRIPTION = """
Replaces empty masks in a mask batch with the previous mask.
"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }
    
    def replaceemptywithlastmasks(self, masks=None):
        out_masks_list = []
        out_masks_list.append(masks[0])

        # Process masks if provided
        if masks is not None:
            for i in range(1,len(masks)):
                non_zero_num = np.count_nonzero(np.array(masks[i]))
                if non_zero_num > 0:
                    out_masks_list.append(masks[i])
                else:
                    out_masks_list.append(out_masks_list[i-1])                  
        out_masks = torch.stack(out_masks_list, dim=0)
        return (out_masks,)