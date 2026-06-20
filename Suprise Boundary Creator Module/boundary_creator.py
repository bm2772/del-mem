import torch
from typing import Tuple, List, Optional
from similarity_refinement import events_with_similarity_adjustment

class SurpriseBoundaryPipeline:
    """
    A standalone pipeline to isolate the surprise function of creating boundaries
    and the graph-theoretic boundary refinement process. This can be integrated
    into new architectures like Qwen.
    """
    
    @staticmethod
    def compute_surprisal(logits: torch.Tensor, em_labels: torch.Tensor) -> torch.Tensor:
        """
        Computes token surprisal from next-token logits and target labels.
        """
        if em_labels.dtype != torch.bool:
            prob = torch.softmax(logits, dim=-1)
            # prob shape: (batch, seq_len, vocab_size)
            # em_labels shape: (batch, seq_len)
            surprisal = -torch.log(torch.gather(prob, dim=-1, index=em_labels.unsqueeze(-1))).squeeze(-1)
            return surprisal
        return em_labels

    @staticmethod
    def compute_boundaries(
        exc_length: int,
        surprisal: torch.Tensor,
        global_remainder_surprisal: torch.Tensor,
        global_remainder_ed: int,
        n_local: int,
        n_init: int,
        surprisal_threshold_gamma: float,
        uniform_blocks: bool = False,
        max_block_size: int = 128
    ) -> torch.Tensor:
        """
        Computes boolean boundary indicators based on the surprise threshold over historical context.
        """
        if uniform_blocks:
            divide = torch.zeros(surprisal.shape, dtype=torch.bool, device=surprisal.device)
            divide[:, ::max_block_size] = True
            return divide
            
        if global_remainder_ed <= exc_length:
            # Not enough history, compute mean/std on the current chunk itself
            std = torch.std(surprisal, dim=-1)
            mean = torch.mean(surprisal, dim=-1)
            divide = surprisal > surprisal_threshold_gamma * std.unsqueeze(-1) + mean.unsqueeze(-1)
        else:
            # Use historical context for stable mean/std
            avg_st = max(global_remainder_ed - n_local, 0)
            avg_ed = max(global_remainder_ed - exc_length, n_init)
            
            history = global_remainder_surprisal[:, avg_st:avg_ed]
            std = torch.std(history, dim=-1)
            mean = torch.mean(history, dim=-1)
            divide = surprisal > surprisal_threshold_gamma * std.unsqueeze(-1) + mean.unsqueeze(-1)
            
        return divide

    @staticmethod
    def refine_boundaries(
        divide: torch.Tensor,
        exc_length: int,
        global_remainder_ed: int,
        global_remainder_len: int,
        global_block_divide: torch.Tensor,
        K_states_for_refinement: List[torch.Tensor],
        refine_with_buffer: bool,
        similarity_metric: str,
        min_block_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Refines the initial surprise-based boundaries using a graph-theoretic approach
        based on cosine similarity of Key states across top layers.
        
        Args:
            K_states_for_refinement: List of key tensors from layer st_layer to the last layer.
                                     Each tensor should be of shape (batch, num_heads, full_seq_len, dim_head)
        """
        batch_size = divide.shape[0]
        
        if global_remainder_len >= 2 * exc_length and refine_with_buffer:
            last_divide = torch.zeros(batch_size)
            for u in range(batch_size):
                last_events = torch.where(global_block_divide[u, global_remainder_ed - 2 * exc_length : global_remainder_ed - exc_length] > 0)[0]
                if len(last_events) == 0:
                    last_divide[u] = exc_length
                else:
                    last_divide[u] = last_events[-1]
            offsets = exc_length - last_divide.int()
            max_offset = torch.max(offsets).item()
        else:
            offsets = torch.zeros(batch_size, dtype=torch.int)
            max_offset = 0

        # Build K matrix from layers
        # Extract the relevant sequence slice from the end of the uncommitted buffer
        slice_len = exc_length + max_offset
        
        # K shape will be (batch, num_layers, num_heads, time, dim_head)
        # We need to clone and move to device
        K = torch.clone(K_states_for_refinement[0][:, :, -slice_len:, :]).unsqueeze(dim=1).to(device)
        for l in range(1, len(K_states_for_refinement)):
            layer_k = torch.clone(K_states_for_refinement[l][:, :, -slice_len:, :]).unsqueeze(dim=1).to(device)
            K = torch.cat((K, layer_k), dim=1)

        # Compute adjacency matrix Stacked A
        # blhtd: batch, layer, head, time, dim
        stacked_A = torch.einsum('blhtd,blhTd->btT', K, K).detach()
        del K
        
        try:
            stacked_A = stacked_A.to(device)
        except Exception as e:
            print(f'Tried casting stacked_A to device, but failed with error: {e}')

        for u in range(batch_size):
            events_sur = torch.where(divide[u] > 0)[0]
            if len(events_sur) > 0:
                off_u = offsets[u].item()
                A_u = stacked_A[u][max_offset - off_u:, max_offset - off_u:]
                
                events_sur_mod = events_with_similarity_adjustment(
                    events_sur, 
                    A_u, 
                    similarity_metric=similarity_metric, 
                    min_size=min_block_size, 
                    offset=off_u
                )
                
                divide[u] = torch.zeros_like(divide[u])
                divide[u][events_sur_mod] = True
                
        del stacked_A
        return divide

class StatefulSurpriseBoundary:
    """
    A stateful wrapper around SurpriseBoundaryPipeline for easier integration into new architectures.
    This class maintains its own history buffers for surprisal and boundaries.
    """
    def __init__(
        self,
        batch_size: int,
        surprisal_threshold_gamma: float = 1.1,
        n_local: int = 4096,
        n_init: int = 128,
        uniform_blocks: bool = False,
        max_block_size: int = 128,
        similarity_refinement: bool = False,
        refine_with_buffer: bool = True,
        similarity_metric: str = 'modularity',
        min_block_size: int = 8,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        self.batch_size = batch_size
        self.surprisal_threshold_gamma = surprisal_threshold_gamma
        self.n_local = n_local
        self.n_init = n_init
        self.uniform_blocks = uniform_blocks
        self.max_block_size = max_block_size
        self.similarity_refinement = similarity_refinement
        self.refine_with_buffer = refine_with_buffer
        self.similarity_metric = similarity_metric
        self.min_block_size = min_block_size
        self.device = device

        self.global_remainder_surprisal = torch.empty((batch_size, 0), dtype=dtype, device=device)
        self.global_block_divide = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
        self.global_remainder_ed = 0
        self.global_remainder_st = 0

    def step(self, logits: torch.Tensor, em_labels: torch.Tensor, K_states_for_refinement: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Process a new chunk of logits to compute boundaries, automatically updating history buffers.
        Returns a boolean tensor of boundaries for the current chunk.
        """
        exc_length = em_labels.shape[-1]
        
        surprisal = SurpriseBoundaryPipeline.compute_surprisal(logits, em_labels)
        
        new_global_remainder_ed = self.global_remainder_ed + exc_length
        
        divide = SurpriseBoundaryPipeline.compute_boundaries(
            exc_length=exc_length,
            surprisal=surprisal,
            global_remainder_surprisal=self.global_remainder_surprisal,
            global_remainder_ed=new_global_remainder_ed,
            n_local=self.n_local,
            n_init=self.n_init,
            surprisal_threshold_gamma=self.surprisal_threshold_gamma,
            uniform_blocks=self.uniform_blocks,
            max_block_size=self.max_block_size
        )
        
        if self.similarity_refinement and K_states_for_refinement is not None:
            global_remainder_len = new_global_remainder_ed - self.global_remainder_st
            divide = SurpriseBoundaryPipeline.refine_boundaries(
                divide=divide,
                exc_length=exc_length,
                global_remainder_ed=new_global_remainder_ed,
                global_remainder_len=global_remainder_len,
                global_block_divide=self.global_block_divide,
                K_states_for_refinement=K_states_for_refinement,
                refine_with_buffer=self.refine_with_buffer,
                similarity_metric=self.similarity_metric,
                min_block_size=self.min_block_size,
                device=self.device
            )
        
        self.global_remainder_ed = new_global_remainder_ed
        if surprisal.dtype == torch.bool:
             self.global_remainder_surprisal = torch.cat((self.global_remainder_surprisal, surprisal.float()), dim=-1)
        else:
             self.global_remainder_surprisal = torch.cat((self.global_remainder_surprisal, surprisal), dim=-1)
             
        self.global_block_divide = torch.cat((self.global_block_divide, divide), dim=-1)
        
        return divide
        
    def shift_history(self, num_tokens_to_remove: int):
        """
        Shift the internal history buffers after memory blocks are successfully committed,
        preventing infinite accumulation of history.
        """
        self.global_remainder_st += num_tokens_to_remove
        self.global_remainder_surprisal = self.global_remainder_surprisal[:, num_tokens_to_remove:]
        self.global_block_divide = self.global_block_divide[:, num_tokens_to_remove:]
