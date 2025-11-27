import torch.nn as nn

class faster_rcnn(nn.Module):
    def __init__(self, backbone, rpn, roi_head):
        super(faster_rcnn, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_head = roi_head

    def forward(self, images, targets=None):
        # Extract features using the backbone
        features = self.backbone(images)

        # Generate proposals using RPN
        proposals, rpn_losses = self.rpn(images, features, targets)

        # Classify and refine proposals using ROI head
        detections, roi_losses = self.roi_head(features, proposals, images.image_sizes, targets)

        # Combine losses if in training mode
        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        return detections