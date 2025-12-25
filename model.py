import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DenseUNetClassifier(nn.Module):
    def __init__(self, n_classes_seg=3, n_classes_cls=3):
        super(DenseUNetClassifier, self).__init__()

        # Backbone: DenseNet121
        densenet = models.densenet121(pretrained=True)
        self.encoder = densenet.features  # Output: [B, 1024, 7, 7]

        # --- U-Net Decoder ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.segmentation_head = nn.Sequential(
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Conv2d(64, n_classes_seg, kernel_size=1)
        )

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # [B, 1024, 1, 1]
            nn.Flatten(),                # [B, 1024]
            nn.Linear(1024, 256),        # [B, 256]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes_cls)  # [B, 3]
        )

    def forward(self, x):
        features = self.encoder(x)  # [B, 1024, 7, 7]

        # Segmentation branch
        seg = F.relu(self.up1(features))
        seg = F.relu(self.up2(seg))
        seg = F.relu(self.up3(seg))
        seg = F.relu(self.up4(seg))
        seg_out = self.segmentation_head(seg)  # [B, 3, 224, 224]

        # Classification branch
        cls_out = self.classifier(features)    # [B, 3]

        return seg_out, cls_out
