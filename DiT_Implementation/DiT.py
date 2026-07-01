import torch
import torch.nn as nn
import math

# ==========================================
# 1. TIMESTEP / CONDITIONING EMBEDDER
# ==========================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(), # SiLU is the standard activation for diffusion models
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ==========================================
# 2. ADAPTIVE LAYER NORM (adaLN)
# ==========================================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ==========================================
# 3. CORE DIT BLOCK (TRANSFORMER BLOCK)
# ==========================================
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out

        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


# ==========================================
# 4. PATCHIFICATION (IMAGE/SPECTROGRAM TO TOKENS)
# ==========================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):

        x = self.proj(x) 
        x = x.flatten(2) 
        x = x.transpose(1, 2)
        return x


# ==========================================
# 5. FULL DIFFUSION TRANSFORMER
# ==========================================
class DiT(nn.Module):

    def __init__(self, img_size=32, patch_size=4, in_channels=4, hidden_size=256, depth=4, num_heads=4):
        super().__init__()

        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)
        )
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.img_size = img_size

    def unpatchify(self, x):
        c = self.in_channels
        p = self.patch_size
        h = w = self.img_size // p
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t):
        x = self.x_embedder(x) + self.pos_embed

        c = self.t_embedder(t)
        

        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x)
        x = self.unpatchify(x)
        
        return x

# ==========================================
# TEST THE MODEL (Dry Run)
# ==========================================
if __name__ == "__main__":
    batch_size = 2
    channels = 4
    spatial_size = 32 

    dummy_noisy_data = torch.randn(batch_size, channels, spatial_size, spatial_size)

    dummy_timesteps = torch.tensor([10, 999])

    print("Initializing Educational DiT...")
    model = DiT(
        img_size=spatial_size, 
        patch_size=4, 
        in_channels=channels, 
        hidden_size=256, 
        depth=6,      
        num_heads=8
    )
    
    print(f"Input shape: {dummy_noisy_data.shape}")
    predicted_noise = model(dummy_noisy_data, dummy_timesteps)
    print(f"Output shape: {predicted_noise.shape}")

    assert predicted_noise.shape == dummy_noisy_data.shape, "Output shape mismatch!"
    print("Success! The DiT successfully processed the data and predicted the noise/flow.")