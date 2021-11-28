import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature=1.0, world_size=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def distance(self, z):
        N = z.size(0)
        dist_mat = torch.zeros((N, N), dtype=torch.float)
        for i in range(N):
            for j in range(N):
                a, b = z[i], z[j]
                dist_mat[i][j] = torch.norm(a-b, dim=-1).mean()

        return dist_mat

    def forward(self, z_i1, z_i2, z_j1, z_j2):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        if z_i1.size(2) * z_i1.size(3) > z_i2.size(2) * z_i2.size(3):
            z_i2 = nn.functional.interpolate(z_i2, (z_i1.size(2), z_i1.size(3)), mode='bilinear', align_corners=True)
            z_j2 = nn.functional.interpolate(z_j2, (z_i1.size(2), z_i1.size(3)), mode='bilinear', align_corners=True)
        else:
            z_i1 = nn.functional.interpolate(z_i1, (z_i2.size(2), z_i2.size(3)), mode='bilinear', align_corners=True)
            z_j1 = nn.functional.interpolate(z_j1, (z_i2.size(2), z_i2.size(3)), mode='bilinear', align_corners=True)

        z_i = torch.cat((z_i1, z_i2), dim=0)
        z_j = torch.cat((z_j1, z_j2), dim=0)

        B, C, H, W = z_i.size()
        N = 2 * self.batch_size * self.world_size

        z_i = z_i.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        z_j = z_j.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        z = torch.cat((z_i, z_j), dim=0)
        z = F.softmax(z, dim=-1)

        #sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        #euclidean_dist = torch.cdist(z, z, p=2) 
        euclidean_dist = self.distance(z)
        sim = torch.exp(-euclidean_dist)

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

if __name__ == '__main__':
    nt_xent = NT_Xent(batch_size=2, temperature=1.0, world_size=1)
    z_i1 = torch.rand(1, 19, 12, 12)
    z_i2 = torch.rand(1, 19, 8, 8)
    z_j1 = torch.rand(1, 19, 12, 12)
    z_j2 = torch.rand(1, 19, 8, 8)
    loss = nt_xent.forward(z_i1, z_i2, z_j1, z_j2)
    print(loss)