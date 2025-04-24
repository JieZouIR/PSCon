import torch
import torch.nn.functional as F

# Build a one-hot mapping matrix from sequence indices
# Args:
#   map_sequence: input sequence tensor of shape [batch_size, seq_len]
#   vocab_size: size of vocabulary
# Returns:
#   b_map_: one-hot encoded tensor of shape [batch_size, seq_len, vocab_size]
def build_map(map_sequence, vocab_size=None):
    batch_size, seq_len = map_sequence.size()
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, seq_len, vocab_size).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, seq_len, vocab_size)
    b_map_.scatter_(2, map_sequence.unsqueeze(2), 1.)
    b_map_.requires_grad=False
    return b_map_

# Calculate pairwise binary cross entropy loss between two sequences
# Args:
#   x: first input tensor
#   y: second input tensor
#   reduction: reduction method for the loss ('mean' or 'sum')
# Returns:
#   loss value
def pairPSCon_binary_cross_entropy_with_logits(x, y, reduction='mean'):
    x=x.unsqueeze(-1).expand(-1, -1, x.size(1))
    y=y.unsqueeze(-1).expand(-1, -1, y.size(1))
    x= x - x.transpose(1, 2)
    y= y - y.transpose(1, 2)

    return F.soft_margin_loss(x, y, reduction=reduction)

# Constants for numerical stability
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

# Get a near-negative infinity value for a given dtype
# Args:
#   dtype: torch data type
# Returns:
#   A finite number close to negative infinity for the given dtype
def neginf(dtype):
    """Return a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

# Generate a square subsequent mask for attention mechanism
# Args:
#   sz: size of the square mask
#   random: whether to randomly mask some positions
# Returns:
#   mask: attention mask tensor of shape [sz, sz]
def generate_square_subsequent_mask(sz, random=False):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(~mask, neginf(torch.float32)).masked_fill(mask, float(0.0))
    if torch.cuda.is_available():
        mask = mask.cuda()
    if random:
        mask.masked_fill_(torch.randn_like(mask)>0.8, neginf(torch.float32))
    return mask
