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
    Memory-efficient Supervised Contrastive Loss for single-label classification.

    Based on SupContrast (https://arxiv.org/abs/2004.11362) and MoCo implementations.
    This loss encourages features of the same class to be closer and features of different classes to be farther apart.
    """

    def __init__(self, temperature=0.07, use_vq=False, vq_num_embeddings=256, vq_embedding_dim=None, vq_commitment_cost=0.25,
                 use_queue=False, queue_size=65536, dim=None):
        super().__init__()
        self.temperature = temperature
        self.use_vq = use_vq
        self.use_queue = use_queue
        self.dim = dim  # Will be set on first forward pass if None

        if self.use_vq:
            self.vq_embedding_dim = vq_embedding_dim  # Will be set on first forward pass if None
            self.vq_num_embeddings = vq_num_embeddings
            self.vq_commitment_cost = vq_commitment_cost
            # VQ layer will be created on first forward pass

        if self.use_queue:
            # Queue will be initialized on first forward pass when we know the dimension
            self.queue_size = queue_size

    def forward(self, features, labels):
        """
        Args:
            features: Feature vectors from the model (batch_size, feature_dim)
            labels: Ground truth labels (batch_size,)

        Returns:
            contrastive_loss: scalar
            vq_loss: scalar (0 if not using VQ)
        """
        # Initialize dimensions on first forward pass
        if self.dim is None:
            self.dim = features.shape[1]
            if self.use_vq and self.vq_embedding_dim is None:
                self.vq_embedding_dim = self.dim
                self.vq_layer = VectorQuantizer(self.vq_num_embeddings, self.vq_embedding_dim, self.vq_commitment_cost)
            if self.use_queue:
                self.register_buffer("queue", torch.randn(self.dim, self.queue_size))
                self.queue = F.normalize(self.queue, dim=0)
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Normalize features for cosine similarity
        features = F.normalize(features, dim=1)

        # Apply VQ if enabled
        vq_loss = 0.0
        if self.use_vq:
            features, vq_loss, _ = self.vq_layer(features)
            # Re-normalize after quantization
            features = F.normalize(features, dim=1)

        # Compute contrastive loss
        if self.use_queue:
            contrastive_loss = self._moco_contrastive_loss(features, labels)
        else:
            contrastive_loss = self._memory_efficient_contrastive_loss(features, labels)

        return contrastive_loss, vq_loss

    def _memory_efficient_contrastive_loss(self, features, labels):
        """
        Memory-efficient supervised contrastive loss based on SupContrast implementation.
        Avoids computing full similarity matrix for better memory usage.
        """
        device = features.device
        batch_size = features.shape[0]

        # Create labels tensor for comparison
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute logits (similarity matrix divided by temperature)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # For numerical stability (subtract max for each anchor)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _moco_contrastive_loss(self, features, labels):
        """
        MoCo-style contrastive loss using queue mechanism for memory efficiency.
        """
        # This is a simplified version - full MoCo implementation would need
        # momentum encoder and distributed training setup
        batch_size = features.shape[0]

        # Positive logits: features with themselves (simplified)
        l_pos = torch.einsum('nc,nc->n', [features, features]).unsqueeze(-1)

        # Negative logits: features with queue
        l_neg = torch.einsum('nc,ck->nk', [features, self.queue.clone().detach()])

        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # Labels: positive keys are the first
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        # Update queue (simplified - in real MoCo this happens after distributed gathering)
        self._dequeue_and_enqueue(features)

        loss = F.cross_entropy(logits, labels)
        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue with new keys (MoCo style)"""
        if not self.use_queue:
            return

        keys = keys.detach()
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue.shape[1] % batch_size == 0  # for simplicity

        # Replace the keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue.shape[1]

        self.queue_ptr[0] = ptr
