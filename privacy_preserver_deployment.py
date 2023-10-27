import torch

import unet_model_v2
import timeit


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Encoder_raw = unet_model_v2.UNetEncoder(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4).to(device)
Decoder = unet_model_v2.UNetDecoder(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4).to(device)

model = 'efficientnet_avg_face.pt'
Encoder_raw.load_state_dict(torch.load("/home/walter/Gaze_privacy/GazeCaptureRes18EncRawMeanFace/"+model))
Decoder.load_state_dict(torch.load("/home/walter/Gaze_privacy/GazeCaptureRes18DecTwoMeanFace/"+model))
anchor_mid = [torch.rand((1, 32, 224, 224)).to(device), torch.rand((1, 64, 112, 112)).to(device),
              torch.rand((1, 128, 56, 56)).to(device), torch.rand((1, 256, 28, 28)).to(device)]
X = torch.rand((1,3,224,224)).to(device)

start = timeit.default_timer()
for i in range(1000):
    representation, _ = Encoder_raw(X)
    reconstructed_img = Decoder(representation, anchor_mid)

stop = timeit.default_timer()

print('Time: ', (stop - start)/1000)
