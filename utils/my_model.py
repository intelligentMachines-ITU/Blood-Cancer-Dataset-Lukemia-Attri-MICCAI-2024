import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler

torch.backends.cudnn.benchmark = False  # You can set it to True if you experience performance gains
torch.backends.cudnn.deterministic = False
from src.loss_functions.losses import AsymmetricLoss, ASLSingleLabel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self, num_classes=12, dropout_prob=0.2, in_channels=3):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 *3* 3, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
    
    def forward(self, x_input):
        # Apply convolutional and pooling layers
        x = F.leaky_relu(self.conv1(x_input))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x))

        # Apply fully connected layers
    
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Rest of the code remains unchanged

# Initialize the model
cell_attribute_model = MyCNN(num_classes=12, dropout_prob=0.5, in_channels=256).to(device)
cell_attribute_model.train()  # Set the model in training mode

# Initialize optimizer, criterion, and scheduler
optimizer_cell_model = torch.optim.SGD(cell_attribute_model.parameters(), lr=0.01, weight_decay=0.01)
step_size = 5
gamma = 0.1
scheduler_cell_model = lr_scheduler.StepLR(optimizer_cell_model, step_size=step_size, gamma=gamma)
#criterion = nn.CrossEntropyLoss()
criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.08, disable_torch_grad_focal_loss=True)
# criterion = ASLSingleLabel()


# /num_classes = 2
#criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

def cell_training(cell_attribute_model_main,cell_datas, labels):
    obj_batch_size = len(cell_datas)
     # Set the model in training mode
    #optimizer_cell_model.zero_grad()

        # Filter out instances with label=2 and their corresponding cell_datas
    # Filter out rows where any element in the row (excluding the first column) is equal to 2
    valid_indices = [i for i, row in enumerate(labels[:,1:]) if not torch.any(row[1:] == 2).item()]
    
    if not valid_indices:
        # print("No valid instances, skipping training.")
        object_batch_loss = torch.tensor(0.0, requires_grad=True, device=device)  # Initialize as a torch.Tensor

        return  object_batch_loss

    filtered_cell_datas = [cell_datas[i] for i in valid_indices]
    filtered_labels = labels[:,1:][valid_indices]

    # Assuming each element in filtered_cell_datas is a tensor of shape (in_channels, height, width)
    cell_images = torch.stack(filtered_cell_datas).to(device)
    cell_datas_batch = cell_images.squeeze(1)
    filtered_labels = filtered_labels.to(device)

    # Initialize the model with the dynamically determined in_channels
    # in_channels = filtered_cell_datas[0].size(1)  # Assuming the first element in filtered_cell_datas defines in_channels
    # cell_attribute_model_main.conv1.in_channels = in_channels

    # Forward pass
    outputs_my = cell_attribute_model_main(cell_datas_batch.float())
    outputs_my = outputs_my.view(len(valid_indices), -1)

    # Process labels to create target_tensor
    # label_att = filtered_labels[:, 5].float()  # Assuming label[5] contains 0 or 1
    # target_tensor = label_att.view(-1, 1)

    # Compute the loss
    num_classes = 2
    one_hot_encoded_tensors = []

    # Perform one-hot encoding for each column separately
    for i in range(filtered_labels.size(1)):
        # Extract the current column
        column_values = filtered_labels[:, i].long()

        # Generate one-hot encoded tensor for the current column
        one_hot_encoded_col = torch.eye(num_classes, device=filtered_labels.device)[column_values]

        # Reshape to match the original shape
        one_hot_encoded_col = one_hot_encoded_col.unsqueeze(1)

        one_hot_encoded_tensors.append(one_hot_encoded_col)

    # Concatenate the one-hot encoded tensors along the second dimension (axis=1)
    one_hot_encoded_result = torch.cat(one_hot_encoded_tensors, dim=1)
    outputs_my = outputs_my.view(outputs_my.size(0), 6,2)

    object_batch_loss = criterion(outputs_my, one_hot_encoded_result)

    # Check if the loss contains NaN
    if torch.isnan(object_batch_loss):
        # If NaN, trigger a breakpoint to inspect variables
        breakpoint()

    torch.use_deterministic_algorithms(False, warn_only=True)

    # Backward pass and optimization
    object_batch_loss = object_batch_loss/len(filtered_labels)
   # object_batch_loss.backward(retain_graph=True)
   # optimizer_cell_model.step()
    #scheduler_cell_model.step()

    # Explicitly release tensors
    #del cell_images, target_tensor
    #torch.cuda.empty_cache()

    return object_batch_loss

