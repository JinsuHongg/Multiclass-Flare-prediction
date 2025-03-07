import torch

# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn


class Alexnet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(Alexnet, self).__init__()

        # Load pretrained AlexNet
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "alexnet", pretrained=True
        )

        # Modify classifier to include MC Dropout
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),  # Apply dropout here
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4),  # Change to match your output classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty


class Mobilenet(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(Mobilenet, self).__init__()

        # Load mobilenet v3
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v3_large", pretrained=True
        )
        self.model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout),  # Apply MC Dropout
            nn.Linear(1280, 4),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty


class Resnet18(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(Resnet18, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )
        # Modify the fully connected (FC) layer with dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),  # Apply MC Dropout
            nn.Linear(512, 4),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty


class Resnet34(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(Resnet34, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet34", pretrained=True
        )
        # Modify the fully connected (FC) layer with dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),  # Apply MC Dropout
            nn.Linear(512, 4),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty


class Resnet50(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(Resnet50, self).__init__()

        # load pretrained architecture from pytorch
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet50", pretrained=True
        )
        # Modify the fully connected (FC) layer with dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),  # Apply MC Dropout
            nn.Linear(512, 4),  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def mc_forward(self, x: torch.Tensor, mc_samples: int = 50) -> torch.Tensor:
        """
        Perform multiple stochastic forward passes using MC Dropout.
        Args:
            x (torch.Tensor): Input image tensor
            mc_samples (int): Number of Monte Carlo samples
        Returns:
            torch.Tensor: Mean prediction over MC samples
        """
        self.train()  # Keep dropout active
        preds = torch.stack([self.forward(x) for _ in range(mc_samples)])
        return preds.mean(dim=0), preds.std(dim=0)  # Mean and uncertainty
