import torch
import torch.nn as nn

MAX_BATCH_SIZE = 64


def get_padding(kernel_size, stride=1, dilation=1, mode="centered"):
    """
    Computes 'same' padding given a kernel size, stride an dilation.

    Parameters
    ----------

    kernel_size: int
        kernel_size of the convolution

    stride: int
        stride of the convolution

    dilation: int
        dilation of the convolution

    mode: str
        either "centered", "causal" or "anticausal"
    """
    if kernel_size == 1: return (0, 0)
    p = (kernel_size - 1) * dilation + 1
    half_p = p // 2
    if mode == "centered":
        p_right = p // 2
        p_left = (p - 1) // 2
    elif mode == "causal":
        p_right = 0
        p_left = p // 2 + (p - 1) // 2
    elif mode == "anticausal":
        p_right = p // 2 + (p - 1) // 2
        p_left = 0
    else:
        raise Exception(f"Padding mode {mode} is not valid")
    return (p_left, p_right)


def get_padding_2d(kernel_size, stride=(1, 1), dilation=(1, 1), mode=("centered", "centered")):
    """
    Computes 2D padding for spectrograms with separate modes for height (frequency) and width (time).
    
    Parameters
    ----------
    kernel_size: tuple of int
        (height, width) kernel size of the 2D convolution
        where height=frequency, width=time
    
    stride: tuple of int
        (height, width) stride of the convolution
        where height=frequency, width=time
    
    dilation: tuple of int
        (height, width) dilation of the convolution
        where height=frequency, width=time
    
    mode: tuple of str
        (height_mode, width_mode) padding modes for each dimension
        each can be "centered", "causal" or "anticausal"
        height_mode typically "centered" for frequency
        width_mode typically "causal" for time
        
    Returns
    -------
    tuple: ((h_left, h_right), (w_left, w_right))
        Padding for height (frequency) and width (time) dimensions
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(mode, str):
        mode = (mode, mode)
    
    h_pad = get_padding(kernel_size[0], stride[0], dilation[0], mode[0])  # frequency
    w_pad = get_padding(kernel_size[1], stride[1], dilation[1], mode[1])  # time
    
    return (h_pad, w_pad)


class CachedSequential(nn.Sequential):
    """
    Sequential operations with future compensation tracking
    """

    def __init__(self, *args, **kwargs):
        cumulative_delay = kwargs.pop("cumulative_delay", 0)
        stride = kwargs.pop("stride", 1)
        super().__init__(*args, **kwargs)

        self.cumulative_delay = int(cumulative_delay) * stride

        last_delay = 0
        for i in range(1, len(self) + 1):
            try:
                last_delay = self[-i].cumulative_delay
                break
            except AttributeError:
                pass
        self.cumulative_delay += last_delay


class Sequential(CachedSequential):
    pass


class CachedPadding1d(nn.Module):
    """
    Cached Padding implementation, replace zero padding with the end of
    the previous tensor.
    """

    def __init__(self, padding, crop=False):
        super().__init__()
        self.initialized = 0
        self.padding = padding
        self.crop = crop

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, _ = x.shape
        self.register_buffer(
            "pad",
            torch.zeros(MAX_BATCH_SIZE, c, self.padding).to(x))
        self.initialized += 1

    def forward(self, x):
        if not self.initialized:
            self.init_cache(x)

        if self.padding:
            x = torch.cat([self.pad[:x.shape[0]], x], -1)
            self.pad[:x.shape[0]].copy_(x[..., -self.padding:])

            if self.crop:
                x = x[..., :-self.padding]

        return x


class CachedPadding2d(nn.Module):
    """
    Cached Padding implementation for 2D tensors (e.g., spectrograms).
    Replaces zero padding with cached values from previous tensors.
    Supports different padding modes for height (frequency) and width (time) dimensions.
    
    Tensor shape convention: (batch_size, channels, frequency_bins, time_frames)
    - Height (dim 2): Frequency dimension - typically uses centered padding
    - Width (dim 3): Time dimension - typically uses causal padding for streaming
    """

    def __init__(self, padding, crop=(False, False)):
        """
        Parameters
        ----------
        padding: tuple of tuples
            ((h_left, h_right), (w_left, w_right)) padding for height (frequency) and width (time)
        crop: tuple of bool
            (crop_h, crop_w) whether to crop each dimension after padding
        """
        super().__init__()
        self.initialized = 0
        
        if isinstance(padding, int):
            self.h_padding = (padding, padding)
            self.w_padding = (padding, padding)
        elif isinstance(padding, tuple) and len(padding) == 2:
            if isinstance(padding[0], int):
                # Assume (h_pad, w_pad) format
                self.h_padding = (padding[0], padding[0])
                self.w_padding = (padding[1], padding[1])
            else:
                # Assume ((h_left, h_right), (w_left, w_right)) format
                self.h_padding = padding[0]
                self.w_padding = padding[1]
        else:
            raise ValueError(f"Invalid padding format: {padding}")
            
        if isinstance(crop, bool):
            self.crop_h = crop
            self.crop_w = crop
        else:
            self.crop_h = crop[0]
            self.crop_w = crop[1]

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, h, w = x.shape
        
        # Cache for height (frequency) padding - only if we have left padding
        if self.h_padding[0] > 0:
            self.register_buffer(
                "h_pad",
                torch.zeros(MAX_BATCH_SIZE, c, self.h_padding[0], w).to(x))
        
        # Cache for width (time) padding - only if we have left padding
        if self.w_padding[0] > 0:
            self.register_buffer(
                "w_pad",
                torch.zeros(MAX_BATCH_SIZE, c, h + sum(self.h_padding), self.w_padding[0]).to(x))
            
        self.initialized += 1

    def forward(self, x):
        if not self.initialized:
            self.init_cache(x)

        b, c, h, w = x.shape
        
        # Apply height (frequency) padding first
        if sum(self.h_padding) > 0:
            pad_top = self.h_padding[0]
            pad_bottom = self.h_padding[1]
            
            if pad_top > 0:
                # Use cached values for left (top) padding
                x = torch.cat([self.h_pad[:b], x], dim=2)
                # Update cache with the rightmost (bottom) values
                self.h_pad[:b].copy_(x[:, :, -pad_top:, :])
                
            if pad_bottom > 0:
                # For frequency dimension, we can use symmetric padding
                x = torch.cat([x, torch.zeros(b, c, pad_bottom, w, device=x.device)], dim=2)
                
            if self.crop_h:
                x = x[:, :, :-sum(self.h_padding), :]
        
        # Apply width (time) padding
        if sum(self.w_padding) > 0:
            pad_left = self.w_padding[0]
            pad_right = self.w_padding[1]
            
            if pad_left > 0:
                # Use cached values for left (past) padding in time dimension
                current_h = x.shape[2]
                if hasattr(self, 'w_pad') and self.w_pad.shape[2] != current_h:
                    # Resize cache if height changed
                    self.w_pad = torch.zeros(MAX_BATCH_SIZE, c, current_h, pad_left, device=x.device)
                
                x = torch.cat([self.w_pad[:b, :, :current_h, :], x], dim=3)
                # Update cache with the rightmost (most recent) values
                self.w_pad[:b, :, :current_h, :].copy_(x[:, :, :, -pad_left:])
                
            if pad_right > 0:
                # Zero padding for right (future) - this maintains causality in time
                x = torch.cat([x, torch.zeros(b, c, x.shape[2], pad_right, device=x.device)], dim=3)
                
            if self.crop_w:
                x = x[:, :, :, :-sum(self.w_padding)]

        return x


class CachedConv1d(nn.Conv1d):
    """
    Implementation of a Conv1d operation with cached padding
    """

    def __init__(self, *args, **kwargs):
        padding = kwargs.get("padding", 0)
        cumulative_delay = kwargs.pop("cumulative_delay", 0)

        kwargs["padding"] = 0

        super().__init__(*args, **kwargs)

        if isinstance(padding, int):
            r_pad = padding
            padding = 2 * padding
        elif isinstance(padding, list) or isinstance(padding, tuple):
            r_pad = padding[1]
            padding = padding[0] + padding[1]

        s = self.stride[0]
        cd = cumulative_delay

        stride_delay = (s - ((r_pad + cd) % s)) % s

        self.cumulative_delay = (r_pad + stride_delay + cd) // s

        self.cache = CachedPadding1d(padding)
        self.downsampling_delay = CachedPadding1d(stride_delay, crop=True)

    def forward(self, x):
        x = self.downsampling_delay(x)
        x = self.cache(x)
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CachedConvTranspose1d(nn.ConvTranspose1d):
    """
    Implementation of a ConvTranspose1d operation with cached padding
    """

    def __init__(self, *args, **kwargs):
        cd = kwargs.pop("cumulative_delay", 0)
        super().__init__(*args, **kwargs)
        stride = self.stride[0]
        self.initialized = 0
        self.cumulative_delay = self.padding[0] + cd * stride

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, _ = x.shape
        self.register_buffer(
            "cache",
            torch.zeros(
                MAX_BATCH_SIZE,
                c,
                2 * self.padding[0],
            ).to(x))
        self.initialized += 1

    def forward(self, x):
        x = nn.functional.conv_transpose1d(
            x,
            self.weight,
            None,
            self.stride,
            0,
            self.output_padding,
            self.groups,
            self.dilation,
        )

        if not self.initialized:
            self.init_cache(x)

        padding = 2 * self.padding[0]

        x[..., :padding] += self.cache[:x.shape[0]]
        self.cache[:x.shape[0]].copy_(x[..., -padding:])

        x = x[..., :-padding]

        bias = self.bias
        if bias is not None:
            x = x + bias.unsqueeze(-1)
        return x


class CachedConvTranspose2d(nn.ConvTranspose2d):
    """
    Implementation of a ConvTranspose2d operation with cached padding for spectrograms.
    Maintains causality in the time dimension (width) while allowing full access to frequency information (height).
    
    Tensor shape convention: (batch_size, channels, frequency_bins, time_frames)
    - Height (dim 2): Frequency dimension
    - Width (dim 3): Time dimension - maintains causality for streaming
    """

    def __init__(self, *args, **kwargs):
        cd = kwargs.pop("cumulative_delay", 0)
        super().__init__(*args, **kwargs)
        stride_h, stride_w = self.stride
        self.initialized = 0
        
        # Calculate cumulative delay primarily from width (time) dimension
        padding_w = self.padding[1] if isinstance(self.padding, (tuple, list)) else self.padding
        self.cumulative_delay = padding_w + cd * stride_w

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, h, w = x.shape
        padding_h = self.padding[0] if isinstance(self.padding, (tuple, list)) else self.padding
        padding_w = self.padding[1] if isinstance(self.padding, (tuple, list)) else self.padding
        
        self.register_buffer(
            "cache",
            torch.zeros(
                MAX_BATCH_SIZE,
                c,
                2 * padding_h,
                2 * padding_w,
            ).to(x))
        self.initialized += 1

    def forward(self, x):
        x = nn.functional.conv_transpose2d(
            x,
            self.weight,
            None,
            self.stride,
            0,
            self.output_padding,
            self.groups,
            self.dilation,
        )

        if not self.initialized:
            self.init_cache(x)

        padding_h = self.padding[0] if isinstance(self.padding, (tuple, list)) else self.padding
        padding_w = self.padding[1] if isinstance(self.padding, (tuple, list)) else self.padding
        
        h_pad = 2 * padding_h
        w_pad = 2 * padding_w

        # Add cached values to the beginning of the tensor
        x[:, :, :h_pad, :w_pad] += self.cache[:x.shape[0]]
        
        # Update cache with the end values
        self.cache[:x.shape[0]].copy_(x[:, :, -h_pad:, -w_pad:])

        # Remove the cached portion from the end
        x = x[:, :, :-h_pad, :-w_pad]

        bias = self.bias
        if bias is not None:
            x = x + bias.unsqueeze(-1).unsqueeze(-1)
        return x


class ConvTranspose1d(nn.ConvTranspose1d):

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("cumulative_delay", 0)
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0


class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("cumulative_delay", 0)
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0


class Conv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        self._pad = kwargs.get("padding", (0, 0))
        kwargs.pop("cumulative_delay", 0)
        kwargs["padding"] = 0

        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0

    def forward(self, x):
        x = nn.functional.pad(x, self._pad)
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        self._pad = kwargs.get("padding", (0, 0, 0, 0))
        kwargs.pop("cumulative_delay", 0)
        kwargs["padding"] = 0

        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0

    def forward(self, x):
        # Handle different padding formats
        if isinstance(self._pad, int):
            pad = (self._pad, self._pad, self._pad, self._pad)
        elif isinstance(self._pad, (tuple, list)) and len(self._pad) == 2:
            pad = (self._pad[1], self._pad[1], self._pad[0], self._pad[0])  # (w_left, w_right, h_top, h_bottom)
        else:
            pad = self._pad
            
        x = nn.functional.pad(x, pad)
        return nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class AlignBranches(nn.Module):

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = list(map(lambda x: x.cumulative_delay, self.branches))

        max_delay = max(delays)

        self.paddings = nn.ModuleList([
            CachedPadding1d(p, crop=True)
            for p in map(lambda f: max_delay - f, delays)
        ])

        self.cumulative_delay = int(cumulative_delay * stride) + max_delay

    def forward(self, x):
        outs = []
        for branch, pad in zip(self.branches, self.paddings):
            delayed_x = pad(x)
            outs.append(branch(delayed_x))
        return outs


class AlignBranches2d(nn.Module):
    """
    Aligns multiple branches with different delays for 2D tensors (spectrograms).
    Primarily handles alignment in the time dimension (width).
    
    Tensor shape convention: (batch_size, channels, frequency_bins, time_frames)
    - Height (dim 2): Frequency dimension
    - Width (dim 3): Time dimension - where alignment/delay matters
    """

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = list(map(lambda x: x.cumulative_delay, self.branches))

        max_delay = max(delays)

        self.paddings = nn.ModuleList([
            CachedPadding2d(((0, 0), (p, 0)), crop=(False, True))
            for p in map(lambda f: max_delay - f, delays)
        ])

        self.cumulative_delay = int(cumulative_delay * stride) + max_delay

    def forward(self, x):
        outs = []
        for branch, pad in zip(self.branches, self.paddings):
            delayed_x = pad(x)
            outs.append(branch(delayed_x))
        return outs


class Branches(nn.Module):

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = list(map(lambda x: x.cumulative_delay, self.branches))

        max_delay = max(delays)

        self.cumulative_delay = int(cumulative_delay * stride) + max_delay

    def forward(self, x):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        return outs


class Branches2d(nn.Module):
    """
    Parallel branches for 2D tensors (spectrograms) without alignment.
    Tracks cumulative delay primarily from the time dimension (width).
    
    Tensor shape convention: (batch_size, channels, frequency_bins, time_frames)
    - Height (dim 2): Frequency dimension
    - Width (dim 3): Time dimension - where delay tracking matters
    """

    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = list(map(lambda x: x.cumulative_delay, self.branches))

        max_delay = max(delays)

        self.cumulative_delay = int(cumulative_delay * stride) + max_delay

    def forward(self, x):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        return outs


class CachedConv2d(nn.Conv2d):
    """
    Implementation of a Conv2d operation with cached padding for spectrograms.
    Maintains causality in the time dimension (width) while allowing full access to frequency information (height).
    
    Tensor shape convention: (batch_size, channels, frequency_bins, time_frames)
    - Height (dim 2): Frequency dimension - typically uses centered padding
    - Width (dim 3): Time dimension - typically uses causal padding for streaming
    """

    def __init__(self, *args, **kwargs):
        padding = kwargs.get("padding", 0)
        cumulative_delay = kwargs.pop("cumulative_delay", 0)
        padding_mode = kwargs.pop("padding_mode", ("centered", "causal"))

        kwargs["padding"] = 0

        super().__init__(*args, **kwargs)

        # Handle different padding input formats
        if isinstance(padding, int):
            h_pad = w_pad = padding
        elif isinstance(padding, (list, tuple)) and len(padding) == 2:
            h_pad, w_pad = padding
        elif isinstance(padding, (list, tuple)) and len(padding) == 4:
            # (pad_left, pad_right, pad_top, pad_bottom) format
            w_pad = (padding[0], padding[1])  # time (left=past, right=future)
            h_pad = (padding[2], padding[3])  # frequency (top, bottom)
        else:
            h_pad = w_pad = 0

        # Ensure padding_mode is a tuple
        if isinstance(padding_mode, str):
            padding_mode = (padding_mode, padding_mode)

        # Calculate padding for each dimension
        kernel_h, kernel_w = self.kernel_size  # (frequency, time)
        stride_h, stride_w = self.stride       # (frequency, time)
        dilation_h, dilation_w = self.dilation # (frequency, time)

        if isinstance(h_pad, int):
            h_padding = get_padding(kernel_h, stride_h, dilation_h, padding_mode[0])  # frequency
            if h_pad != 0:
                # Scale the calculated padding
                scale = h_pad / max(1, sum(h_padding))
                h_padding = (int(h_padding[0] * scale), int(h_padding[1] * scale))
        else:
            h_padding = h_pad

        if isinstance(w_pad, int):
            w_padding = get_padding(kernel_w, stride_w, dilation_w, padding_mode[1])  # time
            if w_pad != 0:
                # Scale the calculated padding
                scale = w_pad / max(1, sum(w_padding))
                w_padding = (int(w_padding[0] * scale), int(w_padding[1] * scale))
        else:
            w_padding = w_pad

        # Calculate cumulative delay (primarily from width/time dimension)
        cd = cumulative_delay
        s_w = stride_w
        r_pad_w = w_padding[1] if isinstance(w_padding, tuple) else w_padding

        stride_delay = (s_w - ((r_pad_w + cd) % s_w)) % s_w
        self.cumulative_delay = (r_pad_w + stride_delay + cd) // s_w

        # Create padding layers
        self.cache = CachedPadding2d((h_padding, w_padding))
        self.downsampling_delay = CachedPadding2d(((0, 0), (stride_delay, 0)), crop=(False, True))

    def forward(self, x):
        x = self.downsampling_delay(x)
        x = self.cache(x)
        return nn.functional.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
