
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
import numpy as np

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
        elif bimamba_type == "v3":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

            A_c = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_c_log = torch.log(A_c)  # Keep A_b_log in fp32
            self.A_c_log = nn.Parameter(A_c_log)
            self.A_c_log._no_weight_decay = True

            self.conv1d_c = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_c = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_c = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_c._no_weight_decay = True

            A_c_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_c_b_log = torch.log(A_c_b)  # Keep A_b_log in fp32
            self.A_c_b_log = nn.Parameter(A_c_b_log)
            self.A_c_b_log._no_weight_decay = True

            self.conv1d_c_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_c_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_c_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_c_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_c_b._no_weight_decay = True

        elif bimamba_type == "v4":
            # --- v4: Initialize parameters for the remaining 5 directions ---
            # Grid size is fixed to (16, 16, 4) for 256x256x64 input with 16x16x16 patches
            # Corresponding to dimensions in 5D view (B, C, H=16, W=16, D=4):
            # R/Height -> dim -3, A/Width -> dim -2, D/Depth -> dim -1

            # 2nd Direction: Reverse along R (Height) - corresponds to dim=-3
            A_r_rev = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_r_rev_log = torch.log(A_r_rev)
            self.A_r_rev_log = nn.Parameter(A_r_rev_log)
            self.A_r_rev_log._no_weight_decay = True
            self.conv1d_r_rev = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.x_proj_r_rev = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_r_rev = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.D_r_rev = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_r_rev._no_weight_decay = True

            # 3rd Direction: Forward along A (Width) - corresponds to dim=-2
            A_a_fwd = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_a_fwd_log = torch.log(A_a_fwd)
            self.A_a_fwd_log = nn.Parameter(A_a_fwd_log)
            self.A_a_fwd_log._no_weight_decay = True
            self.conv1d_a_fwd = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.x_proj_a_fwd = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_a_fwd = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.D_a_fwd = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_a_fwd._no_weight_decay = True

            # 4th Direction: Reverse along A (Width) - corresponds to dim=-2
            A_a_rev = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_a_rev_log = torch.log(A_a_rev)
            self.A_a_rev_log = nn.Parameter(A_a_rev_log)
            self.A_a_rev_log._no_weight_decay = True
            self.conv1d_a_rev = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.x_proj_a_rev = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_a_rev = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.D_a_rev = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_a_rev._no_weight_decay = True

            # 5th Direction: Forward along D (Depth) - corresponds to dim=-1
            A_d_fwd = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_d_fwd_log = torch.log(A_d_fwd)
            self.A_d_fwd_log = nn.Parameter(A_d_fwd_log)
            self.A_d_fwd_log._no_weight_decay = True
            self.conv1d_d_fwd = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.x_proj_d_fwd = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_d_fwd = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.D_d_fwd = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_d_fwd._no_weight_decay = True

            # 6th Direction: Reverse along D (Depth) - corresponds to dim=-1
            A_d_rev = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_d_rev_log = torch.log(A_d_rev)
            self.A_d_rev_log = nn.Parameter(A_d_rev_log)
            self.A_d_rev_log._no_weight_decay = True
            self.conv1d_d_rev = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.x_proj_d_rev = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_d_rev = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.D_d_rev = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_d_rev._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    def forward(self, hidden_states, prompt: Optional[Tensor] = None, inference_params=None):
        """
        hidden_states: (B, L, D)
        prompt: (B, L, D_inner) - 注意维度是 d_inner
        Returns: same shape as hidden_states
        """
        if prompt is not None:
            if prompt.dtype != hidden_states.dtype:
                prompt = prompt.to(hidden_states.dtype)
                
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        # xz  B D L
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                out = bimamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )    
            elif self.bimamba_type == "v3":

                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )
                B, D, L = xz.shape
                token_position = L//2
                cls, xc = (xz[:, :, token_position:token_position+1],
                           torch.cat([xz[:, :, :token_position], xz[:, :, token_position+1:]], dim=-1))
                xc = xc.reshape(B, D, int(np.sqrt(L)), int(np.sqrt(L)))
                xc = xc.permute(0,1,3,2).reshape(B, D, -1)
                xc = torch.cat((xc[:, :, :token_position], cls, xc[:, :, token_position:]), dim=-1)
                A_c = -torch.exp(self.A_c_log.float())
                out_c = mamba_inner_fn_no_out_proj(
                    xc,
                    self.conv1d_c.weight,
                    self.conv1d_c.bias,
                    self.x_proj_c.weight,
                    self.dt_proj_c.weight,
                    A_c,
                    None,
                    None,
                    self.D_c.float(),
                    delta_bias=self.dt_proj_c.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )

                A_c_b = -torch.exp(self.A_c_b_log.float())
                out_c_b = mamba_inner_fn_no_out_proj(
                    xc.flip([-1]),
                    self.conv1d_c_b.weight,
                    self.conv1d_c_b.bias,
                    self.x_proj_c_b.weight,
                    self.dt_proj_c_b.weight,
                    A_c_b,
                    None,
                    None,
                    self.D_c_b.float(),
                    delta_bias=self.dt_proj_c_b.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )
                #print(xz.mean(), out.mean(), out_b.mean(), out_c.mean(), out_c_b.mean())
                # if torch.isinf(out.mean()) or torch.isnan(out.mean()):
                #     out = out_b.flip([-1])
                # if torch.isinf(out_b.mean()) or torch.isnan(out_b.mean()):
                #     out_b = out.flip([-1])
                # if torch.isinf(out_c.mean()) or torch.isnan(out_c.mean()):
                #     out_c = out_c_b.flip([-1])
                # if torch.isinf(out_c_b.mean()) or torch.isnan(out_c_b.mean()):
                #     out_c_b = out_c.flip([-1])
                # print(xz.mean(), out.mean(), out_b.mean(), out_c.mean(), out_c_b.mean())
                out_c = out_c + out_c_b.flip([-1])
                cls, out_c = (out_c[:, :, token_position:token_position + 1],
                           torch.cat([out_c[:, :, :token_position], out_c[:, :, token_position + 1:]], dim=-1))
                out_c = out_c.reshape(B, self.d_inner, int(np.sqrt(L)), int(np.sqrt(L)))
                out_c = out_c.permute(0, 1, 3, 2).reshape(B, self.d_inner, -1)
                out_c = torch.cat((out_c[:, :, :token_position], cls, out_c[:, :, token_position:]), dim=-1)
                out = out + out_b.flip([-1])
                out = F.linear(rearrange((out+out_c)/4.,
                                         "b d l -> b l d"),
                               self.out_proj.weight,
                               self.out_proj.bias)

            elif self.bimamba_type == "v4":
                # --- v4: 6-directional scanning Forward ---
                grid_size = (16, 16, 4)
                B, D_, L = xz.shape

                # ===============================================================
                #                      START: 变量名修改区域
                # ===============================================================

                # 1. Standard Forward scan (+R / H-axis) 
                # 这里默认扫描是沿 H 轴的，所以我们使用 H 轴的参数（即默认参数）
                out_h_fwd = mamba_inner_fn_no_out_proj( # <-- 修改 out_d -> out_h_fwd
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )

                # 2. Reverse R scan (-R / H-axis)
                A_h_bwd = -torch.exp(self.A_r_rev_log.float()) # <-- 修改 A_d_b -> A_h_bwd
                out_h_bwd = mamba_inner_fn_no_out_proj( # <-- 修改 out_d_b -> out_h_bwd
                    xz.flip([-1]),
                    self.conv1d_r_rev.weight,
                    self.conv1d_r_rev.bias,
                    self.x_proj_r_rev.weight,
                    self.dt_proj_r_rev.weight,
                    A_h_bwd, # <-- 修改
                    None,
                    None,
                    self.D_r_rev.float(),
                    delta_bias=self.dt_proj_r_rev.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )


                # 3. Forward A scan (+A / W-axis)
                xz_w = xz.view(B, D_, *grid_size).permute(0, 1, 2, 4, 3).reshape(B, D_, L) # (B, D_, H, D, W) -> # <-- 修改 xz_a -> xz_w
                A_w_fwd = -torch.exp(self.A_a_fwd_log.float()) # <-- 修改 A_a -> A_w_fwd
                out_w_fwd = mamba_inner_fn_no_out_proj( # <-- 修改 out_a -> out_w_fwd
                    xz_w, # <-- 修改
                    self.conv1d_a_fwd.weight,
                    self.conv1d_a_fwd.bias,
                    self.x_proj_a_fwd.weight,
                    self.dt_proj_a_fwd.weight,
                    A_w_fwd, # <-- 修改
                    None,
                    None,
                    self.D_a_fwd.float(),
                    delta_bias=self.dt_proj_a_fwd.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )

                # 4. Reverse A scan (-A / W-axis)
                A_w_bwd = -torch.exp(self.A_a_rev_log.float()) # <-- 修改 A_a_b -> A_w_bwd
                out_w_bwd = mamba_inner_fn_no_out_proj( # <-- 修改 out_a_b -> out_w_bwd
                    xz_w.flip([-1]), # <-- 修改 xz_a -> xz_w
                    self.conv1d_a_rev.weight,
                    self.conv1d_a_rev.bias,
                    self.x_proj_a_rev.weight,
                    self.dt_proj_a_rev.weight,
                    A_w_bwd, # <-- 修改
                    None,
                    None,
                    self.D_a_rev.float(),
                    delta_bias=self.dt_proj_a_rev.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )

                # 5. Forward D scan (+D / D-axis)
                xz_d = xz.view(B, D_, *grid_size).permute(0, 1, 4, 3, 2).reshape(B, D_, L) # (B, D_, D, W, H) -> # <-- 修改 xz_r -> xz_d
                A_d_fwd = -torch.exp(self.A_d_fwd_log.float()) # <-- 修改 A_r -> A_d_fwd
                out_d_fwd = mamba_inner_fn_no_out_proj( # <-- 修改 out_r -> out_d_fwd
                    xz_d, # <-- 修改
                    self.conv1d_d_fwd.weight,
                    self.conv1d_d_fwd.bias,
                    self.x_proj_d_fwd.weight,
                    self.dt_proj_d_fwd.weight,
                    A_d_fwd, # <-- 修改
                    None,
                    None,
                    self.D_d_fwd.float(),
                    delta_bias=self.dt_proj_d_fwd.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )

                # 6. Reverse D scan (-D / D-axis)
                A_d_bwd = -torch.exp(self.A_d_rev_log.float()) # <-- 修改 A_r_b -> A_d_bwd
                out_d_bwd = mamba_inner_fn_no_out_proj( # <-- 修改 out_r_b -> out_d_bwd
                    xz_d.flip([-1]), # <-- 修改 xz_r -> xz_d
                    self.conv1d_d_rev.weight,
                    self.conv1d_d_rev.bias,
                    self.x_proj_d_rev.weight,
                    self.dt_proj_d_rev.weight,
                    A_d_bwd, # <-- 修改
                    None,
                    None,
                    self.D_d_rev.float(),
                    delta_bias=self.dt_proj_d_rev.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )
                
                # 结果恢复与融合
                out_d = out_d_fwd.reshape(B, self.d_inner, 4, 16, 16).permute(0, 1, 4, 3, 2).reshape(B, self.d_inner, L) + out_d_bwd.flip([-1]).reshape(B, self.d_inner, 4, 16, 16).permute(0, 1, 4, 3, 2).reshape(B, self.d_inner, L) # <-- 修改 out_r -> out_d, out_r_b -> out_d_bwd

                out_w = out_w_fwd.reshape(B, self.d_inner, 16, 4, 16).permute(0, 1, 2, 4, 3).reshape(B, self.d_inner, L) + out_w_bwd.flip([-1]).reshape(B, self.d_inner, 16, 4, 16).permute(0, 1, 2, 4, 3).reshape(B, self.d_inner, L) # <-- 修改 out_a -> out_w, out_a_b -> out_w_bwd
                
                out_h = out_h_fwd + out_h_bwd.flip([-1]) # <-- 修改 out_d -> out_h_fwd, out_d_b -> out_h_bwd

                # 7. Aggregate outputs (average)
                out_combined = out_d + out_w + out_h # <-- 修改 out_r -> out_d, out_a -> out_w, out_d -> out_h

                # ===============================================================
                #                        END: 变量名修改区域
                # ===============================================================

                # 8. Apply final output projection
                out = F.linear(rearrange(out_combined, "b d l -> b l d"),
                               self.out_proj.weight,
                               self.out_proj.bias)
                # --- v4 Forward 结束 ---



            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    prompt=prompt,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        if self.init_layer_scale is not None:
                out = out * self.gamma    
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
