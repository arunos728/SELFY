import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ImportError:
    print('[ImportError] Cannot import SpatialCorrelationSampler')


class STSSTransformation(nn.Module):
    def __init__(self, d_in, d_hid, num_segments, window=(5,9,9), use_corr_sampler=True):
        super(STSSTransformation, self).__init__()
        self.num_segments = num_segments
        self.window = window
        assert window[1] == window[2]
        self.use_corr_sampler = use_corr_sampler
        
        if use_corr_sampler:
            try:
                self.correlation_sampler = SpatialCorrelationSampler(1, window[1], 1, 0, 1)
            except:
                print("[Warning] SpatialCorrelationSampler cannot be used.")
                self.use_corr_sampler = False
            
        # Resize spatial resolution to 14x14
        self.downsample = nn.Sequential(
            nn.Conv2d(d_in, d_hid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_hid),
            nn.ReLU(inplace=True)
        )
        
    def _L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        
        return (x / norm)

    def _corr_abs_to_rel(self, corr, h, w):
        # Naive implementation of spatial correlation sampler
        
        max_d = self.window[1] // 2

        b,c,s = corr.size()        
        corr = corr.view(b,h,w,h,w)

        w_diag = tr.zeros((b,h,h,self.window[1],w),device='cuda')
        for i in range(max_d+1):
            if (i==0):
                w_corr_offset = tr.diagonal(corr, offset=0, dim1=2, dim2=4)       
                w_diag[:,:,:,max_d] = w_corr_offset
            else:
                w_corr_offset_pos = tr.diagonal(corr, offset=i, dim1=2, dim2=4) 
                w_corr_offset_pos = F.pad(w_corr_offset_pos, (i,0))
                w_diag[:,:,:,max_d-i] = w_corr_offset_pos
                w_corr_offset_neg = tr.diagonal(corr, offset=-i, dim1=2, dim2=4) 
                w_corr_offset_neg = F.pad(w_corr_offset_neg, (0,i))
                w_diag[:,:,:,max_d+i] = w_corr_offset_neg

        hw_diag = tr.zeros((b,self.window[1],w,self.window[1],h), device='cuda') 
        for i in range(max_d+1):
            if (i==0):
                h_corr_offset = tr.diagonal(w_diag, offset=0, dim1=1, dim2=2)
                hw_diag[:,:,:,max_d] = h_corr_offset
            else:
                h_corr_offset_pos = tr.diagonal(w_diag, offset=i, dim1=1, dim2=2) 
                h_corr_offset_pos = F.pad(h_corr_offset_pos, (i,0))
                hw_diag[:,:,:,max_d-i] = h_corr_offset_pos
                h_corr_offset_neg = tr.diagonal(w_diag, offset=-i,dim1=1, dim2=2) 
                h_corr_offset_neg = F.pad(h_corr_offset_neg, (0,i))     
                hw_diag[:,:,:,max_d+i] = h_corr_offset_neg 

        hw_diag = hw_diag.permute(0,3,1,4,2).contiguous()
        hw_diag = hw_diag.view(-1, self.window[1], self.window[1], h, w)      

        return hw_diag         

    def _correlation(self, feature1, feature2):
        feature1 = self._L2normalize(feature1) # btl, c, h, w
        feature2 = self._L2normalize(feature2) # btl, c, h, w

        if self.use_corr_sampler:
            corr = self.correlation_sampler(feature1, feature2)
        else:
            b, c, h, w = feature1.size()
            feature1 = rearrange(feature1, 'b c h w -> b c (h w)')
            feature2 = rearrange(feature2, 'b c h w -> b c (h w)')
            corr = tr.einsum('bcn,bcm->bnm',feature2, feature1)
            corr = self._corr_abs_to_rel(corr, h, w)

        return corr
        
    def forward(self, x):
        # resize spatial resolution to 14x14
        x = self.downsample(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
        
        x_pre = repeat(x, 'b t c h w -> (b t l) c h w', l=self.window[0])
        x_post = F.pad(x, (0,0,0,0,0,0,self.window[0]//2,self.window[0]//2), 'constant', 0).unfold(1,self.window[0],1)
        x_post = rearrange(x_post, 'b t c h w l -> (b t l) c h w')     
        
        stss = self._correlation(x_pre, x_post)   
        stss = rearrange(stss, '(b t l) u v h w -> b t h w 1 l u v', t=self.num_segments, l=self.window[0])
        
        return stss

    
class STSSExtraction(nn.Module):
    def __init__(self, num_segments, window=(5,9,9), chnls=(4,16,64,64)):
        super(STSSExtraction, self).__init__()
        self.num_segments = num_segments
        self.window = window
        self.chnls = chnls
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(1, chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[1]),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(chnls[1], chnls[2], kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[2]),
            nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(chnls[2], chnls[3], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,0,0), bias=False),
            nn.BatchNorm3d(chnls[3]),
            nn.ReLU(inplace=True))    
        
    def forward(self, x):
        b,t,h,w,_,l,u,v = x.size()
        x = rearrange(x, 'b t h w 1 l u v -> (b t h w) 1 l u v')
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = rearrange(x, '(b t h w) c l 1 1 -> (b l) c t h w', t=t, h=h, w=w)
        
        return x
    
    
class STSSIntegration(nn.Module):
    def __init__(self, d_in, d_out, num_segments, window=(5,9,9), chnls=(64,64,64,64)):
        super(STSSIntegration, self).__init__()
        self.num_segments = num_segments
        self.window = window
        self.chnls = chnls
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(d_in, chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[1]),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(chnls[1], chnls[2], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[2]),
            nn.ReLU(inplace=True))
        
        self.conv3_fuse = nn.Sequential(
            Rearrange('(b l) c t h w -> b (l c) t h w', l=self.window[0]),
            nn.Conv3d(chnls[2]*self.window[0], chnls[3], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[3]),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(chnls[3], d_out, kernel_size=1, stride=(1,2,2), padding=(0,0,0), output_padding=(0,1,1), bias=False),
            nn.BatchNorm3d(d_out),
            Rearrange('b c t h w -> (b t) c h w')
        )
        
        
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3_fuse(x)
        x = self.upsample(x)
        
        return x
    
    
    
class SELFYBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid,
                 num_segments=8,
                 window=(5,9,9),
                 ext_chnls=(4,16,64,64),
                 int_chnls=(64,64,64,64)
                ):
        super(SELFYBlock, self).__init__()
        
        self.stss_transformation = STSSTransformation(
            d_in,
            d_hid,
            num_segments=num_segments,
            window=window,
        )
        
        self.stss_extraction = STSSExtraction(
            num_segments=num_segments,
            window = window,
            chnls = ext_chnls
        )
        
        self.stss_integration = STSSIntegration(
            ext_chnls[-1],
            d_in,
            num_segments=num_segments,
            window = window,
            chnls = int_chnls
        )
        
        
    def forward(self, x):
        identity = x
        out = self.stss_transformation(x)
        out = self.stss_extraction(out)
        out = self.stss_integration(out)
        
        out = out + identity
        out = F.relu(out)
        
        return out