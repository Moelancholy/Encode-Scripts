import vapoursynth as vs
from math import ceil

import awsmfunc as awf
import debandshit
import G41Fun as gf
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import mvsfunc as mvf
import placebo
import vardefunc as vdf
import xvs
import zzfunc as zzf

from adptvgrnMod import adptvgrnMod
from functools import partial
from rektlvl import rektlvls
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from vsutil import *

core = vs.core
core.max_cache_size = 1024 * 32

def average_resize(clip, w=None, h=None):
    iw = clip.width
    ih = clip.height
    ow = fallback(w, iw)
    oh = fallback(h, ih)
    
    if (ow > iw and ow/iw != ow//iw) or (oh > ih and oh/ih != oh//ih):
        clip = clip.resize.Point(iw * ceil(ow / iw), ih * ceil(oh / ih))
    
    return clip.fmtc.resample(ow, oh, kernel='box', fulls=1, fulld=1)



def eedi3s(clip, field=0, alpha=0.2, beta=0.5, gamma=40, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out
    
    

def LevelsM(clip, points, levels, xpass=[0, 'peak'], return_expr=False):
    qm = len(points)
    peak = [(1 << clip.format.bits_per_sample) - 1, 1][clip.format.sample_type]
   
    if len(set(xpass)) == 1:
        expr = f'x {points[0]} < x {points[-1]} > or {xpass[0]} '
        qm -= 1
    else:
        expr = f'x {points[0]} < {xpass[0]} x {points[-1]} > {xpass[-1]} '
   
    for x in range(len(points) - 1):
        if points[x+1] < points[-1]:
            expr += f' x {points[x+1]} <= '
        if levels[x] == levels[x+1]:
            expr += f' {peak * levels[x]} '
        else:
            expr += f' x {points[x]} - {peak * (levels[x+1] - levels[x])/(points[x+1] - points[x])} * {peak * levels[x]} + '
   
    for _ in range(qm):
        expr += ' ? '
   
    expr = expr.replace('  ', ' ').replace('peak', f'{peak}')
   
    if return_expr:
        return expr
   
    return clip.std.Expr(expr)
    


def clamp_aa(src: vs.VideoNode, weak: vs.VideoNode, strong: vs.VideoNode, strength: float = 1) -> vs.VideoNode:
    """
    Clamp stronger AAs to weaker AAs.
    Useful for clamping upscaled_sraa or eedi3 to nnedi3 for a strong but precise AA.
    Stolen from Zastin.
    :param src:      Non-AA'd source clip.
    :param weak:     Weakly-AA'd clip (eg: nnedi3)
    :param strong:   Strongly-AA'd clip (eg: eedi3)
    :param strength: Clamping strength (Default: 1)
    :return:         Clip with clamped anti-aliasing.
    """
    if src.format is None or weak.format is None or strong.format is None:
        raise ValueError("nneedi3_clamp: 'Variable-format clips not supported'")
    thr = strength * (1 << (src.format.bits_per_sample - 8)) if src.format.sample_type == vs.INTEGER \
        else strength/219
    clamp = core.std.Expr([get_y(src), get_y(weak), get_y(strong)],
                          expr=f"x y - x z - xor x x y - abs x z - abs < z y {thr} + min y {thr} - max z ? ?"
                          if thr != 0 else "x y z min max y z max min")
    return clamp if src.format.color_family == vs.GRAY \
        else core.std.ShufflePlanes([clamp, src], planes=[0, 1, 2], colorfamily=vs.YUV)

    

def detail_mask(clip: vs.VideoNode,
                sigma: float = 1.0, rxsigma: List[int] = [50, 200, 350],
                pf: bool = False, pf_sigma: float = 1.0,
                brz: Tuple[int] = (2500, 4500), rg_mode: int = 17,
                ) -> vs.VideoNode:
    """
    A detail mask aimed at preserving as much detail as possible
    within darker areas, even if it contains mostly noise.
    """
    bits = get_depth(clip)

    if bits != 16:
        clip = depth(clip, 16)

    clip_y = get_y(clip)
    pf = core.bilateral.Gaussian(clip_y, sigma=pf_sigma) if pf else clip_y
    ret = core.retinex.MSRCP(pf, sigma=rxsigma, upper_thr=0.005)

    blur_ret = core.bilateral.Gaussian(ret, sigma=sigma)
    blur_ret_diff = core.std.Expr([blur_ret, ret], "x y -").std.Binarize(brz[0])

    kirsch = kgf.kirsch(clip_y).std.Binarize(brz[1])
    merged = core.std.Expr([blur_ret_diff, kirsch], "x y +")
    rm_grain = core.rgvs.RemoveGrain(merged, rg_mode)
    return rm_grain if bits == 16 else depth(rm_grain, bits)    




shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"


no_aa_ranges = [(0, 521), (131354, 137123)]


src = core.lsmas.LWLibavSource(r"E:\Anime\[210611] 機動戦士ガンダム 閃光のハサウェイ(劇場限定版)\GUNDAM_HATHAWAY\BDMV\STREAM\00002.m2ts")[48:137172]
src = depth(src, 16)

fb = rektlvls(src, rownum=[0, 1, 1078, 1079], rowval=[4 , 1, 1, 4], colnum=[0, 1, 1918, 1919], colval=[4, 1, 1, 4])
bb = awf.bbmod(fb, top=3, bottom=3, right=3, left=3, thresh=50, blur=600)
bby = get_y(bb)


predenyuv = haf.SMDegrain(bb, tr=2, RefineMotion=True, thSAD=100)
preden = get_y(predenyuv)
preden = vdf.decsiz(preden, sigmaS=4, min_in=208 << 8, max_in=232 << 8)
diff = core.std.MakeDiff(bb, predenyuv)
chromadn = core.knlm.KNLMeansCL(bb, h=0.50, a=3, d=2, s=3, device_type='gpu', device_id=0, channels='UV')
#merge = core.std.MergeDiff(preden, diff)


detail_mask = detail_mask(bby)
ed_gray = core.std.ShufflePlanes(preden, 0, vs.GRAY)
rmask = debandshit.rangemask(ed_gray, 2, 2)
emask = ed_gray.std.Prewitt()

em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
em_hi = iterate(em_hi, core.std.Minimum, 5)
em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
em_me = iterate(em_me, core.std.Minimum, 4)
em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
em_lo = iterate(em_lo, core.std.Minimum, 2)
rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')

weighted = LevelsM(rm_txt, points=[x *256 for x in (1.75, 2.5, 5, 10)], levels=[0, 1, 1, 0], xpass=[0,0], return_expr=0)
texture_mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')
texture_mask2 = core.std.Expr([texture_mask, texture_mask], 'x y +')
texture_mask3 = core.std.Merge(texture_mask, texture_mask2, 0.3)
dbmask = core.std.Expr([detail_mask, texture_mask3], 'x y max')


mask = bby.std.Prewitt().std.Binarize(60 << 8).std.Maximum().rgvs.RemoveGrain(20)
aa_mask = mask.std.Convolution([1]*9)

aa1 = lvf.sraa(preden, rfactor=1.2, downscaler=None, gamma=40)
aa1 = aa1.resize.Bicubic(width=1920, height=1080)
aa1 = preden.std.MaskedMerge(aa1, aa_mask)
aa1 = core.rgvs.Repair(aa1, preden, 13)


upscale = vdf.fsrcnnx_upscale(preden, width = 3840, height = 2160, downscaler = None, shader_file=shader, profile = 'zastin')
aawidth, aaheight = 2240, 1260
aascale = upscale.resize.Bicubic(aawidth, aaheight)
mclip = average_resize(mask, aawidth, aaheight)
fsr = aascale.std.Transpose()

aa2 = eedi3s(get_y(fsr), mclip=mclip.std.Transpose())
aa2 = aa2.std.Transpose()
aa2 = eedi3s(aa2, mclip=mclip)
aa2 = aa2.resize.Bicubic(width=1920, height=1080)
aa2 = preden.std.MaskedMerge(aa2, aa_mask)
aa2 = core.rgvs.Repair(aa2, preden, 13)

aa3 = clamp_aa(preden, aa2, aa1, strength=2)
aa = lvf.rfs(aa3, preden, no_aa_ranges)

halo_m = haf.HQDeringmod(preden, show=True)
dering = haf.EdgeCleaner(aa, 10, smode=1, hot=True, rmode=13)
dehalo = gf.MaskedDHA(dering, rx=2.0, ry=2.0, darkstr=0.00, brightstr=1.0, maskpull=40, maskpush=180)
dh = core.std.MaskedMerge(aa, dehalo, halo_m)

yuv = vdf.merge_chroma(dh, chromadn)
deband = vdf.dumb3kdb(yuv, radius=18, threshold=[26, 30, 30], grain=12)
db = core.std.MaskedMerge(yuv, deband, dbmask)
merge = core.std.MergeDiff(db, diff)


grain = adptvgrnMod(merge, strength=0.10, size=1.20, luma_scaling=12, sharp=60, static=False, lo=19, hi=[200, 240, 240])

out = depth(grain, 10)
out.set_output()