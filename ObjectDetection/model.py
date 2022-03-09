import torchvision.models as models
import torchvision.ops as ops
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def get_instance_segmentation_model(num_classes):
    # load a pre-trained model for classification and return features
    backbone = models.mobilenet_v2(pretrained=True).features
    # FasterRCNN need to know the number of output channels in backbone.
    backbone.out_channels = 1280 # MobilenetV2 = 1280

    # Make RPN generate 4x3 anchors per spatial location,
    # 4 different sizes and 3 different aspect ratio.
    # Tuple[Tuple[int]] ~ each feature map could potentially have
    # different sizes and aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32,64,128,256),),
                        aspect_ratios=((0.5,1.0,2.0),))
    
    # Define what feature maps will be used to perform region of interest
    # cropping, as well as the size of the crop after rescaling
    # If backbone return Tensor -> featmap_names is expected to be [0]
    # elif backbone return OrderedDict[Tensor] -> featmap_names can be chosen 
    # which feature maps to use
    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0'],
                                        output_size=7,
                                        sampling_ratio=2)

    mask_roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0'],
                                            output_size=14,
                                            sampling_ratio=2)    

    model = MaskRCNN(backbone,
                        num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler,
                        mask_roi_pool=mask_roi_pooler)
    return model 
