import vapoursynth as vs
core = vs.core

import awsmfunc as awf
import debandshit as dbs
import kagefunc as kgf
import lvsfunc as lvf
import vardefunc as vdf
import vardefunc.mask as mask

from adptvgrnMod import adptvgrnMod
from functools import partial
from kagefunc import retinex_edgemask
from math import ceil
from havsfunc import SMDegrain, FineDehalo, HQDeringmod
from muvsfunc import SSIM_downsample
from rekt import rektlvls
from thresholding import kimagure_thresholding
from vardefunc.scale import fsrcnnx_upscale, nnedi3cl_double, to_444
from vardefunc.sharp import z4usm
from vsutil import depth, fallback, get_y, get_depth, iterate, get_w, plane



###CUSTOM FUNCS
def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out


###SOURCES
src = depth(core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Do It Yourself!! - 02 (Amazon Prime CBR 1080p).mkv"), 16)
cr = depth(core.lsmas.LWLibavSource(r"E:\Anime\[SubsPlease] Do It Yourself!! - 02 (1080p) [DE0954CA].mkv"), 16)
cr = src[:24] + cr

crmask2 = mask.FreyChen().get_mask(get_y(cr)).std.Binarize(25<<7).std.Convolution([1]*9)
crmasked2 = core.std.MaskedMerge(src, cr, crmask2)
src16 = core.std.ShufflePlanes([crmasked2, cr], [0, 1, 2], vs.YUV)


###AA
y = get_y(src16)
yrgb = core.resize.Bicubic(y, format = vs.RGBS)

double = core.akarin.DLISR(yrgb, scale = 2)
down1 = SSIM_downsample(double, w = 2880, h = 1620, filter_param_a = -1/2, filter_param_b = 1/4)
ydown = core.resize.Bicubic(down1, format = vs.GRAY16, matrix = 1)

aa1 = eedi3s(ydown.std.Transpose(), gamma = 80)
aa1 = eedi3s(aa1.std.Transpose(), gamma = 80)
aa1 = SSIM_downsample(core.resize.Bicubic(aa1, format = vs.RGBS), w = 1920, h = 1080, filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format = vs.GRAY16, matrix = 1)


###DEHALO
double = aa1.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose().znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose()
dehalo = FineDehalo(double, rx = 2.5, darkstr = 0, brightstr = 1).resize.Point(1920, 1080)


###GRAIN
merge1 = core.std.ShufflePlanes([dehalo, src16], [0, 1, 2], vs.YUV)
grainmask1 = plane(HQDeringmod(merge1, mrad = 3, incedge = True, mthr = 60, show = True), 0)
grainmask2 = awf.bandmask(merge1, 420, pix=12, dec=2, exp=2).std.Convolution([1]*9)

grainmask = core.std.Expr([grainmask1, grainmask2], "x y max")
grain = core.grain.Add(merge1, seed = 1, var = 1.69, uvar = 0.0, constant = True)
grain = core.std.MaskedMerge(merge1, grain, grainmask)


###OUTPUT
out = depth(grain, 10)
out.set_output()


#src.set_output(1)
#cr.set_output(2)
#crmask.set_output(3)
#crmasked.set_output(4)
#crmask2.set_output(5)
#crmasked2.set_output(6)
