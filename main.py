from collections import defaultdict
from transformers import AutoImageProcessor, AutoModelForImageClassification
import kagglehub
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os

# from datasets import load_dataset


class FineTunedResnet(nn.Module):
    def __init__(self, model, output_labels, num_classes=10):
        super().__init__()
        self.model = model
        self.output_labels = output_labels
        # Because resnet has 10000 output classes (logits)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(**x).logits
        x = self.classifier(x)
        return x


class TrainingData(Dataset):
    def __init__(self, image_labels):
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image_path, label = self.image_labels[index]
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        return image, label


def train_model(
    processor,
    model,
    criterion,
    optimizer,
    training_dataset,
    device="cpu",
    num_epochs=25,
):
    # for a bunch of epochs (which are iterations until all training data is used)
    # train_dataset = Subset(training_dataset, range(1000))

    for epoch in range(num_epochs):
        training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
        avg_loss = 0
        for images, actual_labels in training_dataloader:

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            predicted_label_logits = model(inputs)
            # predicted_label_indices = predicted_label_logits.argmax(1)
            loss = criterion(predicted_label_logits, actual_labels)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        avg_loss /= len(training_dataloader)
        print(f"Epoch {epoch} loss: {avg_loss}")

    return model

    # for each data point
    # forward pass
    # calculate loss
    # backpropagate
    # update weights
    # in the same epoch,
    pass


def main():
    path = kagglehub.dataset_download("msambare/fer2013")

    image_labels = []
    labels = []
    label_to_index = {}

    index = 0
    for class_name in os.listdir(os.path.join(path, "train")):
        labels.append(class_name)
        if class_name not in label_to_index:
            label_to_index[class_name] = index
            index += 1
        for image_name in os.listdir(os.path.join(path, "train", class_name)):
            image_path = os.path.join(path, "train", class_name, image_name)
            image_labels.append((image_path, label_to_index[class_name]))

    # device = (
    #     torch.accelerator.current_accelerator().type
    #     if torch.accelerator.is_available()
    #     else "cpu"
    # )
    # print(f"Using {device} device")
    device = "cpu"
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    training_data = TrainingData(image_labels)

    resnet = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    model = FineTunedResnet(resnet, labels, num_classes=len(labels))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        processor, model, criterion, optimizer, training_data, device, num_epochs=25
    )

    image = Image.open(path + "/train/angry/Training_10118481.jpg")
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])


if __name__ == "__main__":
    main()
