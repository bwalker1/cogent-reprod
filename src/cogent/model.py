"""Model architecture for Cogent."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from cogent.utils import cosine_distance


class _DecoderOutReshape(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1).transpose(0, 1)


class Model(nn.Module):
    """Cogent model for learning condition-dependent gene representations.

    Args:
        n_genes: Number of genes
        n_groups: Number of groups/conditions
        n_channels: Number of channels (1 for expression only, >1 with spatial)
        embedding_dim: Dimension of gene embeddings (default: 64)
        n_heads: Number of attention heads (default: 4)
        emb_dropout: Dropout for embeddings (default: 0.0)
        decoder_dim: Dimension of decoder layers (default: 64)
        decoder_depth: Number of decoder layers (default: 3)
        cross_p: Cross-gene probability for augmentation (default: 0.05)
        gene_batch_size: Batch size for gene dimension (default: 512)
        beta: Temperature parameter for attention (default: 5.0)
        rand_embeddings: If True, initialize embeddings randomly per group
        initial_embeddings: Optional pre-computed embeddings tensor
        lambda_reg: Regularization coefficient for embedding stability (default: 0.001)
    """

    def __init__(
        self,
        n_genes: int,
        n_groups: int,
        n_channels: int,
        embedding_dim: int = 64,
        n_heads: int = 4,
        emb_dropout: float = 0.0,
        decoder_dim: int = 64,
        decoder_depth: int = 3,
        cross_p: float = 0.05,
        gene_batch_size: int = 512,
        beta: float = 5.0,
        rand_embeddings: bool = False,
        initial_embeddings: torch.Tensor | None = None,
        lambda_reg: float = 0.001,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.n_groups = n_groups
        self.n_channels = n_channels
        self.gene_batch_size = gene_batch_size
        self.cross_p = cross_p
        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        self.range_embedding_dim = embedding_dim // 2
        self.decoder_embedding_dim = embedding_dim // 2
        self.decoder_depth = decoder_depth
        self.eps = 1e-4
        self.final_activation = nn.Identity()
        self.loss = nn.BCEWithLogitsLoss()

        self.expression_attn = CaveBlock(
            heads=n_heads,
            embedding_dim=embedding_dim,
            dropout=emb_dropout,
            n_channels=n_channels,
            gene_batch_size=gene_batch_size,
            beta=beta,
        )

        layers = []
        old_dim = n_heads + self.decoder_embedding_dim
        for layer in range(self.decoder_depth):
            block = nn.Sequential(
                nn.Linear(old_dim, decoder_dim),
                nn.ReLU(),
            )
            old_dim = decoder_dim
            if layer > 0:
                block = ResidualWrapper(block)
            layers.append(block)

        layers.extend(
            [
                nn.Linear(decoder_dim, 1),
                self.final_activation,
                _DecoderOutReshape(),
            ]
        )

        self.decoder_model = nn.Sequential(*layers)

        if initial_embeddings is not None:
            self.embeddings = nn.Parameter(initial_embeddings)
        elif rand_embeddings:
            base_embeddings = torch.randn(
                self.n_groups,
                self.n_genes,
                self.embedding_dim
                + self.range_embedding_dim
                + self.decoder_embedding_dim,
            )
            self.embeddings = nn.Parameter(base_embeddings)
        else:
            base_embeddings = torch.randn(
                1,
                self.n_genes,
                self.embedding_dim
                + self.range_embedding_dim
                + self.decoder_embedding_dim,
            )
            self.embeddings = nn.Parameter(
                torch.cat([base_embeddings for _ in range(self.n_groups)], dim=0)
            )

    def forward(
        self,
        expression: torch.Tensor,
        id: int | torch.Tensor,
        gene_batch_idx: int,
        cross_genes: bool,
        gene_filter: torch.Tensor = None,
        has_spatial: bool = True,
    ):
        """Forward pass with loss computation.

        Args:
            expression: Expression tensor (n_cells, n_genes, n_channels)
            id: Group ID
            gene_batch_idx: Starting index for gene batch
            cross_genes: Whether to apply cross-gene augmentation
            gene_filter: Optional gene filter mask
            has_spatial: Whether this batch contains meaningful spatial
                convolution channels. False for dissociated scRNA-seq cells.

        Returns:
            Tuple of (predictions, loss)
        """
        expr_pred = self.predict(
            expression=expression,
            id=id,
            gene_batch_idx=gene_batch_idx,
            cross_genes=cross_genes,
            gene_filter=gene_filter,
            has_spatial=has_spatial,
        )

        if gene_batch_idx is not None:
            cur_batch_expr = expression[
                :, gene_batch_idx : (gene_batch_idx + self.gene_batch_size), 0
            ]
        else:
            cur_batch_expr = expression[:, :, 0]

        loss = self.loss(expr_pred, cur_batch_expr)
        if self.lambda_reg > 0:
            loss = loss + self.lambda_reg * self.regularization_loss(id=id)

        return expr_pred, loss

    def predict(
        self,
        expression: torch.Tensor,
        id: int | torch.Tensor,
        gene_batch_idx: int,
        cross_genes: bool,
        gene_filter: torch.Tensor = None,
        has_spatial: bool = True,
    ) -> torch.Tensor:
        """Predict expression from embeddings.

        Args:
            expression: Expression tensor (n_cells, n_genes, n_channels)
            id: Group ID
            gene_batch_idx: Starting index for gene batch
            cross_genes: Whether to apply cross-gene augmentation
            gene_filter: Optional gene filter mask

        Returns:
            Predicted expression (n_cells, gene_batch_size)
        """
        key_embeddings = self.get_embedding(id, cross_genes)

        if gene_filter is not None:
            key_embeddings = key_embeddings[gene_filter, :]

        return self.predict_from_embedding(
            expression,
            key_embeddings,
            gene_batch_idx=gene_batch_idx,
            has_spatial=has_spatial,
        )

    def predict_from_embedding(
        self,
        expression: torch.Tensor,
        embeddings: torch.Tensor,
        gene_batch_idx: int,
        has_spatial: bool = True,
    ) -> torch.Tensor:
        """Predict expression from given embeddings.

        Args:
            expression: Expression tensor (n_cells, n_genes, n_channels)
            embeddings: Gene embeddings (n_genes, embedding_dim)
            gene_batch_idx: Starting index for gene batch

        Returns:
            Predicted expression (n_cells, gene_batch_size)
        """
        expr_attn = self.expression_attn(
            embeddings, expression, gene_batch_idx, has_spatial=has_spatial
        )
        cur_batch_embeddings = embeddings[
            gene_batch_idx : (gene_batch_idx + self.gene_batch_size), ...
        ]
        x = cur_batch_embeddings[:, None, -self.decoder_embedding_dim :]
        x = x.expand(-1, expression.shape[0], -1)
        x = torch.cat([x, expr_attn], dim=-1)
        return self.decoder_model(x)

    def get_embedding(self, id: int | torch.Tensor, cross_genes: bool) -> torch.Tensor:
        """Get embeddings for a group with optional cross-gene augmentation.

        Args:
            id: Group ID
            cross_genes: Whether to apply cross-gene augmentation

        Returns:
            Gene embeddings (n_genes, embedding_dim)
        """
        n_groups = self.embeddings.size(0)
        if not cross_genes or n_groups == 1:
            return self.embeddings[
                id.item() if isinstance(id, torch.Tensor) else id, ...
            ]
        else:
            v = torch.squeeze(F.one_hot(id, num_classes=n_groups))
            probs = (1 - self.cross_p) * v + self.cross_p * (1 - v)
            sampled_vals = torch.multinomial(
                probs, self.embeddings.size(1), replacement=True
            )
            s = F.one_hot(sampled_vals, num_classes=n_groups).to(self.embeddings)

        embeddings = torch.einsum("gr, rgd -> gd", s, self.embeddings)
        return embeddings

    def regularization_loss(self, id: int | torch.Tensor) -> torch.Tensor:
        """Compute embedding regularization loss.

        Compare the current group's interaction embeddings with detached
        interaction embeddings for the same genes across all groups.

        Args:
            id: Group ID

        Returns:
            Scalar regularization loss (1 - cosine_similarity)
        """
        all_emb_detached_norm = F.normalize(
            self.embeddings[:, :, : self.embedding_dim].detach(), dim=-1
        )
        cur_emb = self.embeddings.index_select(0, id.reshape(1))[
            :, :, : self.embedding_dim
        ]
        cur_emb_norm = F.normalize(cur_emb, dim=-1)
        regularization_loss = torch.mean(
            1 - torch.einsum("ibd,kbd->kb", cur_emb_norm, all_emb_detached_norm)
        )
        return regularization_loss

    @property
    def device(self):
        """Get device of model parameters."""
        return self.embeddings.device


class CaveBlock(nn.Module):
    """Attention block with spatial awareness.

    Args:
        embedding_dim: Dimension of gene embeddings
        gene_batch_size: Batch size for gene dimension
        heads: Number of attention heads
        dropout: Dropout probability
        n_channels: Number of input channels
        beta: Temperature parameter for attention
    """

    def __init__(
        self,
        embedding_dim: int,
        gene_batch_size: int,
        heads: int = 1,
        dropout: float = 0.0,
        n_channels: int = 1,
        beta: float = 5.0,
    ):
        super().__init__()

        self.has_spatial = n_channels > 1
        self.n_channels = n_channels
        self.gene_batch_size = gene_batch_size
        self.heads = heads
        self.beta = beta
        self.eps = 1e-4
        self.kernel_fn = nn.Sequential(
            nn.Conv1d(1, self.heads, kernel_size=1, bias=False),
            orthogonal(nn.Linear(embedding_dim, embedding_dim, bias=False)),
        )

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.embedding_dim = embedding_dim
        self.range_embedding_dim = embedding_dim // 2
        self.range_model = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(
                    n_channels, self.range_embedding_dim, self.range_embedding_dim
                )
            )
        )

        self.register_buffer(
            "_no_spatial_range",
            torch.ones(1, 1, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_diag_indices",
            torch.arange(gene_batch_size, dtype=torch.long),
            persistent=False,
        )

    def compute_attn(
        self, embeddings: torch.Tensor, gene_batch_idx: int, force_no_spatial: bool
    ) -> torch.Tensor:
        if self.has_spatial and not force_no_spatial:
            embeddings_range = torch.einsum(
                "cde, gd -> gce",
                self.range_model,
                embeddings[
                    :,
                    self.embedding_dim : (
                        self.embedding_dim + self.range_embedding_dim
                    ),
                ],
            )
            q_range = embeddings_range[
                gene_batch_idx : (gene_batch_idx + self.gene_batch_size), :
            ]
            k_range = embeddings_range
            range = torch.einsum("bcd, gcd -> bgc", q_range, k_range)
            range = F.softmax(range, dim=-1)
        else:
            range = self._no_spatial_range.to(embeddings.dtype)

        embeddings_mag = embeddings[:, : self.embedding_dim]
        embeddings_mag = embeddings_mag[:, None, :]
        embeddings_mag = torch.renorm(embeddings_mag, p=2, dim=0, maxnorm=1 - self.eps)
        embeddings_mag = self.kernel_fn(embeddings_mag)
        q_mag = embeddings_mag[
            gene_batch_idx : (gene_batch_idx + self.gene_batch_size), :, :
        ]
        k_mag = embeddings_mag
        dots = -cosine_distance(q_mag, k_mag)
        dots = rearrange(dots, "b h g -> b g h")
        dots = self.beta * dots

        relevant_size = min(self.gene_batch_size, dots.shape[1] - gene_batch_idx)
        diag_indices = self._diag_indices[:relevant_size]
        dots[diag_indices, gene_batch_idx + diag_indices, :] = -100.0
        attn = F.softmax(dots, dim=-2)
        attn = self.dropout(attn)

        attn = torch.einsum("bgh,bgc->bghc", attn, range)

        return attn

    def forward(
        self,
        embeddings: torch.Tensor,
        expression: torch.Tensor,
        gene_batch_idx: int,
        has_spatial: bool = True,
    ) -> torch.Tensor:
        if self.n_channels != 1 and (
            expression.shape[2] == 1 or not self.has_spatial or not has_spatial
        ):
            force_no_spatial = True
            expression = expression[:, :, :1]
        else:
            force_no_spatial = False

        attn = self.compute_attn(
            embeddings, force_no_spatial=force_no_spatial, gene_batch_idx=gene_batch_idx
        )

        out = torch.einsum("bghc,egc->beh", attn, expression)

        return out


class ResidualWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        if not isinstance(module, nn.Module):
            raise ValueError("module must be nn.Module")
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        return self.module(x) + x
