
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import torch

CE = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

BCEL = nn.BCEWithLogitsLoss()

def negative_sampling_contrastive_loss(v1, v2, labels):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  eye = torch.diag_embed(labels).to(v1.device)
  return BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye), logits.diag() > 0


class TripletLoss(nn.Module):
    """Triplet loss class

    Parameters
    ----------
    margin : float
        Ranking loss margin
    metric : string
        Distance metric (either euclidean or cosine)
    """

    def __init__(self, margin=0.3, metric='cosine'):
      super(TripletLoss, self).__init__()
      self.distance_function = euclidean_dist if metric == 'euclidean' else cosine_dist
      self.metric = metric
      self.margin = margin
      self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, im, s):
      # compute text-molecule score matrix
      # batch_size x batch_size
      scores_i2r = self.distance_function(normalize(im, dim=-1),
                                          normalize(s, dim=-1))
      scores_r2i = scores_i2r.t()

      pos = torch.eye(im.size(0))
      neg = 1 - pos

      pos = (pos == 1).to(im.device)
      neg = (neg == 1).to(im.device)

      # positive similarities
      # batch_size x 1
      d1 = scores_i2r.diag().view(im.size(0), 1)
      d2 = d1.t()

      y = torch.ones(scores_i2r.size(0)).to(im.device)

      # text anchor - molecule positive bs x bs
      d1 = d1.expand_as(scores_i2r)
      # molecule anchor - text positive
      # d2 = d2.expand_as(scores_i2r) #bs x bs
      d2 = d2.expand_as(scores_r2i)

      y = y.expand_as(scores_i2r)

      # compare every diagonal score to scores in its column
      # batch_size x batch_size (each anchor is compared to all elements in the batch)
      cost_im = self.ranking_loss(scores_i2r, d1, y)

      # compare every diagonal score to scores in its row
      cost_s = self.ranking_loss(scores_r2i, d2, y)

      # clear diagonals
      cost_s = cost_s.masked_fill_(pos, 0)
      cost_im = cost_im.masked_fill_(pos, 0)

      return (cost_s + cost_im).mean()


def cosine_dist(im, s):
  """Cosine similarity between all the text and molecule pairs
  """

  return 1 - im.mm(s.t())


def euclidean_dist(x, y):
  """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
  """
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
  yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
  dist = xx + yy
  dist.addmm_(1, -2, x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

  return dist




