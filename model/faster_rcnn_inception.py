from faster_rcnn import FasterRCNN

from utils.config import CLASS_NUM


class FasterRCNNInception(FasterRCNN):
    def __init__(self):
        n_fg_classes = CLASS_NUM

        extractor = inception3()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = InceptionROIHEad()