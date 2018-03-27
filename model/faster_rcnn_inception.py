from faster_rcnn import FasterRCNN


class FasterRCNNInception(FasterRCNN):
    def __init__(self):
        n_fg_classes = 1000

        extractor = inception3()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = InceptionROIHEad()