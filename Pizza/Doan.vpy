from turtle import down
import vapoursynth as vs


core = vs.core


import debandshit as dbs
import lvsfunc as lvf
import vardefunc.mask as mask

from adptvgrnMod import adptvgrnMod, sizedgrn
from functools import partial
from kagefunc import retinex_edgemask
from lvsfunc.scale import descale_detail_mask
from havsfunc import SMDegrain
from math import ceil
from muvsfunc import SSIM_downsample
from thresholding import thresholding, Thresholds
from vardefunc.scale import fsrcnnx_upscale, nnedi3cl_double, to_444
from vardefunc.sharp import z4usm
from vsutil import get_depth, get_w, get_y, depth, iterate, fallback


###CUSTOM FUNCS
def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
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


def texture_mask(clip, range):
    ed_gray = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    rmask = mask.MinMax(range).get_mask(clip, lthr=0, multi=1.00)
    emask = ed_gray.std.Prewitt()
    em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
    em_hi = iterate(em_hi, core.std.Minimum, 5)
    em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
    em_me = iterate(em_me, core.std.Minimum, 4)
    em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
    em_lo = iterate(em_lo, core.std.Minimum, 2)
    rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')
    weighted = LevelsM(rm_txt, points=[x *256 for x in (1.75, 2.5, 5, 10)], levels=[0, 1, 1, 0], xpass=[0,0], return_expr=0)

    masked = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')
    return masked



def CCD_denoise(clip: vs.VideoNode, strength):
    depth_clip = get_depth(clip)
    clip1 = to_444(clip, width = clip.width, height = clip.height, join_planes = True)
    clip1 = core.resize.Bicubic(clip1, matrix_in = 1, primaries_in = 1, transfer_in = 1, matrix = 1, primaries = 1, transfer = 1)
    clip1 = core.resize.Bicubic(clip1, format = vs.RGBS)
    clip1 = core.ccd.CCD(clip1, strength)
    clip1 = core.resize.Bicubic(clip1, format = vs.YUV420P16, matrix = 1)
    clip1 = depth(clip1, depth_clip)
    clip = core.std.ShufflePlanes([clip, clip1], [0, 1, 2], vs.YUV)
    return clip



###IMPORT
src = depth(core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][220603][BDOT-0268／BCXA-1757~8][機動戦士ガンダム ククルス・ドアンの島][劇場限定版]\DOANS_ISLAND\BDMV\STREAM\00002.m2ts"),32)
srcy = get_y(src)
shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"

###RESCALE + AA
descale = core.descale.Debicubic(srcy, get_w(810), 810, 0, 1/2)
rescale = nnedi3cl_double(descale, pscrn = 1)

downscale = core.resize.Bicubic(rescale, format = vs.RGBS)
downscale = SSIM_downsample(downscale, 1920, 1080, filter_param_a = -1/2, filter_param_b = 1/4)
downscale = get_y(core.resize.Bicubic(downscale, format = vs.YUV420P16, matrix = 1))

mediumaa = eedi3s(rescale.std.Transpose(), gamma = 80)
mediumaa = eedi3s(mediumaa.std.Transpose(), gamma = 80)
mediumaa = core.resize.Bicubic(mediumaa, format = vs.RGBS)
mediumaa = SSIM_downsample(mediumaa, w = 1920, h = 1080, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4)
aa = get_y(core.resize.Bicubic(mediumaa, format = vs.YUV420P16, matrix = 1))

line_mask = retinex_edgemask(depth(srcy, 16))
aamask = downscale.std.Prewitt().std.Binarize(60 << 8).std.Maximum().rgvs.RemoveGrain(20)
aa_masked = core.std.MaskedMerge(downscale, aa, aamask)
aa_merged = core.std.ShufflePlanes([aa_masked, depth(src, 16)], [0, 1, 2], vs.YUV)


###DENOISE
preden = SMDegrain(depth(aa_masked, 16), tr = 2, thSAD = 100, RefineMotion = True, prefilter = 2)
tmask = texture_mask(preden, range = 1)

cden2 = CCD_denoise(aa_merged, 4.5)
dn_masked = core.std.MaskedMerge(cden2, aa_merged, tmask).std.MaskedMerge(aa_merged, line_mask)


###DEBAND
deband = dbs.dumb3kdb(dn_masked, radius = 17, threshold = [35, 30, 30], grain = 0)
debandgrained = adptvgrnMod(deband, strength = 0.05, static = True, luma_scaling = 4, grain_chroma = False)
debandgrained = adptvgrnMod(debandgrained, strength = 0.15, static = False, luma_scaling = 9, grain_chroma = False)
debandmasked = core.std.MaskedMerge(debandgrained, dn_masked, tmask).std.MaskedMerge(dn_masked, line_mask)


###GRAIN + OUTPUT
grain = adptvgrnMod(debandgrained, strength = 0.04, static = False, luma_scaling = 2, grain_chroma = False)
out = depth(grain, 10)


#src.set_output(1)
#dn_masked.set_output(2)
#debandmasked.set_output(3)
out.set_output()
#debandmasked.set_output(5)


#124043 chroma denoise
#126012, 119182 banding