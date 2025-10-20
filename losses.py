import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE style training.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, embedding_dim)
        
        Returns:
            quantized: (batch_size, embedding_dim)
            vq_loss: scalar
            indices: (batch_size,) - codebook indices
        """
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)

        # VQ Loss
        vq_loss = F.mse_loss(quantized.detach(), flat_input)
        commitment_loss = F.mse_loss(quantized, flat_input.detach())
        vq_loss += self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()

        return quantized, vq_loss, encoding_indices.squeeze()


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

    def get_base_loss(self, outputs, labels):
        """
        Get the base criterion loss without distillation.
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        return self.base_criterion(outputs, labels)


class ContrastiveLoss(torch.nn.Module):
    """
    Supervised Contrastive Loss for single-label classification.
    
    This loss encourages features of the same class to be closer and features of different classes to be farther apart.
    """

    def __init__(self, temperature=0.07, use_vq=False, vq_num_embeddings=256, vq_embedding_dim=256, vq_commitment_cost=0.25):
        super().__init__()
        self.temperature = temperature
        self.use_vq = use_vq
        
        if self.use_vq:
            self.vq_layer = VectorQuantizer(vq_num_embeddings, vq_embedding_dim, vq_commitment_cost)

    def forward(self, features, labels):
        """
        Args:
            features: Feature vectors from the model (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            contrastive_loss: scalar
            vq_loss: scalar (0 if not using VQ)
        """
        # Normalize features for cosine similarity
        features = F.normalize(features, dim=1)
        
        # Apply VQ if enabled
        vq_loss = 0.0
        if self.use_vq:
            features, vq_loss, _ = self.vq_layer(features)
            # Re-normalize after quantization
            features = F.normalize(features, dim=1)
        
        # Compute contrastive loss
        contrastive_loss = self._supervise_contrastive_loss(features, labels)
        
        return contrastive_loss, vq_loss

    def _supervise_contrastive_loss(self, features, labels):
        """
        Compute supervised contrastive loss using NT-Xent style.
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T).float()
        positive_mask.fill_diagonal_(0)  # Remove self-similarity
        
        # Create mask for negative pairs (different class)
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)  # Remove self
        
        # For each sample, compute log probability
        exp_sim = torch.exp(similarity_matrix)
        
        # Numerator: sum of exp similarities for positive pairs
        numerator = exp_sim * positive_mask
        
        # Denominator: sum of exp similarities for all pairs (including positives)
        denominator = exp_sim * (1 - torch.eye(batch_size, device=device))
        
        # Compute log probabilities
        log_prob = torch.log(numerator.sum(dim=1) / denominator.sum(dim=1))
        
        # Only consider samples that have positive pairs
        has_positive = positive_mask.sum(dim=1) > 0
        if has_positive.sum() > 0:
            loss = -log_prob[has_positive].mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
