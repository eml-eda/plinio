import torch
import torch.nn as nn
import torch.nn.functional as F


# Example of 3D CNN
class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_classes = 1
        self.in_channel = 13
        self.filter_size = 3
        self.patch_size = 7

        self.num_filter = 40
        self.num_filter_2 = int(self.num_filter * 3 / 4) + self.num_filter
        self.dilation = 1

        dilation = (self.dilation, 1, 1)

        self.conv1 = nn.Conv3d(
            1,
            self.num_filter,
            (self.filter_size, self.filter_size, self.filter_size),
            stride=(1, 1, 1),
            dilation=dilation,
            padding=0)

        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            self.num_filter,
            self.num_filter,
            (self.filter_size, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            self.num_filter,
            self.num_filter_2,
            (self.filter_size, self.filter_size, self.filter_size),
            dilation=dilation,
            stride=(1, 1, 1),
            padding=(1, 0, 0),
        )
        self.pool2 = nn.Conv3d(
            self.num_filter_2,
            self.num_filter_2,
            (self.filter_size, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            self.num_filter_2,
            self.num_filter_2,
            (self.filter_size, 1, 1),
            dilation=dilation,
            stride=(1, 1, 1),
            padding=(1, 0, 0),
        )
        self.conv4 = nn.Conv3d(
            self.num_filter_2,
            self.num_filter_2,
            (2, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )

        self.features_size = self._get_final_flattened_size()

        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc1 = nn.Linear(self.features_size, 1)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.in_channel, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):   # , embeddings=False): --> DJP: removed to make graph static
        x = torch.unsqueeze(x, 1)  # DJP --> could be moved out of model to simplify plinification
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        # replaced view with flatten for plinification
        # emb = x.view(-1, self.features_size)
        emb = torch.flatten(x, 1)
        x = self.fc1(emb)
        #x= torch.sigmoid(x)
        # commented out to avoid dynamic DNN graph
        # if embeddings:
            # return x, emb
        return x

