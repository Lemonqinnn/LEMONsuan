from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import cv2
import torch
from net import Model

model = Model(scale_cls=7,num_classes=8)
resume = '/home/lemon/few-shot/fewshot-CAN/ChengR/CAN_ResNet_5_5/temp_Gobal4/model_best.pth.tar'
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['state_dict'])

# final_convname = 'clasifier'

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# model._modules.get(final_convname).register_forward_hook(hook_feature)
# print(model.state_dict())
# get the softmax weight
params = list(model.parameters())

weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(
            feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((84,84)),
   transforms.ToTensor(),
   normalize
])
img_path = '/home/lemon/few-shot/DN4/dataset/miniImageNet/mini-imagenet/images/n0153282900000040.jpg'
img_path2 = '/home/lemon/few-shot/DN4/dataset/miniImageNet/mini-imagenet/images/n0153282900000046.jpg'

with open(img_path2,'rb') as f:
    img_pil = Image.open(f)
    img_pil = img_pil.convert('RGB')
    img_pil.save('test2.jpg')
img_tensor_test = preprocess(img_pil)
img_variable_test = img_tensor_test.unsqueeze(0).unsqueeze(0)

with open(img_path,'rb') as f:
    img_pil = Image.open(f)
    img_pil = img_pil.convert('RGB')
    img_pil.save('test.jpg')

img_tensor = preprocess(img_pil)
img_variable = img_tensor.unsqueeze(0).unsqueeze(0)

y_train = torch.from_numpy(np.array([[[0,0,1,0,0]]])).float()
y_test = torch.from_numpy(np.array([[[0,0,1,0,0]]])).float()

model.train()
logit,cls_score,f = model(img_variable,img_variable_test,y_train,y_test)
logit = logit.mean(2)
logit = logit.mean(2)
logit = logit.view(logit.size(0),-1)
img_train = f[0].unsqueeze(0).data.cpu().numpy()
img_test = f[1].unsqueeze(0).data.cpu().numpy()
# print(features_blobs[0].shape)
# download the imagenet category list
# classes = {int(key):value for (key, value)
#           in requests.get(LABELS_URL).json().items()}
# print(len(classes))

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
# for i in range(0, 5):
#     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(img_train, weight_softmax, [idx[0]])

# render the CAM and output
# print('output origianl_.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread('test.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)

CAMs = returnCAM(img_test, weight_softmax, [idx[0]])
img = cv2.imread('test2.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM2.jpg', result)