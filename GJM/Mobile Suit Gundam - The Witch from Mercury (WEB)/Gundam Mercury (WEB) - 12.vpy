import vapoursynth as vs
core = vs.core

core.max_cache_size = 40*1024
core.num_threads = 12

import awsmfunc as awf
import vsdeband as dbs
import vsmasktools as mask

from adptvgrnmod import adptvgrnMod
from havsfunc import SMDegrain, HQDeringmod
from jvsfunc import ccdmod, retinex_edgemask
from lvsfunc import dpir
from vsutil import depth, get_y, plane, iterate
from vsaa import Eedi3SR, Nnedi3SR, masked_clamp_aa
from vsdehalo import fine_dehalo
from vsdenoise import chroma_reconstruct
from vsrgtools import lehmer_diff_merge
from vstools import replace_ranges


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
    rmask = mask.MinMax(range).edgemask(clip, lthr=0, multi=1.00)
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


###SCENEFILTER
edranges = [
    (30113, 32271)
]


###SOURCES
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 15 (Amazon dAnime CBR 1080p).mkv")
vbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 15 (Amazon dAnime VBR 1080p).mkv")
cr = depth(core.lsmas.LWLibavSource(r"B:\[SubsPlease] Mobile Suit Gundam - The Witch from Mercury - 12 (1080p) [4F1F2B64].mkv"), 16)

blank = core.std.BlankClip(cr)
src = blank[:192] + cr
cbr16 = depth(cbr, 16)
ed = cbr16[:192] + src[192:]


###FIX BORDERS
borders = awf.bbmod(ed, left = 1, right = 1, thresh = 5, blur = 999, u = True, v = True)
lehmer = lehmer_diff_merge([cbr, vbr])


###DEBLOCK
deblock1 = depth(dpir(lehmer, 6, cuda = 'trt', tiles = 2, fp16 = True), 16)


###AA
borders_y = get_y(borders)
aa_mask = borders_y.std.Prewitt().std.Binarize(60 << 8).std.Maximum().std.Convolution([1]*9)
aa_masked = masked_clamp_aa(borders_y, 1.5, mask = aa_mask, weak_aa=Nnedi3SR(), strong_aa=Eedi3SR())


###DEHALO
dehalo = fine_dehalo(aa_masked, rx = 2.2, pre_ss = 2, brightstr=0.9)
dh32 = depth(dehalo, 32)


###DENOISE LUMA
preden = depth(SMDegrain(dehalo, tr = 2, thSAD = 120, RefineMotion = True, prefilter = 2), 32)
denoise_y = core.bm3dcuda.BM3D(clip = dh32, ref = preden, sigma = 0.77, radius = 2).bm3d.VAggregate(radius = 2)
denoise_y = core.bm3dcuda.BM3D(clip = dh32, ref = depth(denoise_y, 32), sigma = 0.77, radius = 2).bm3d.VAggregate(radius = 2)


###DERING
merge1 = core.std.ShufflePlanes([denoise_y, borders], [0, 1, 2], vs.YUV)
merge1 = deblock1[:192] + merge1[192:]


###CHROMA
chromadn = ccdmod(merge1, 4)
linemask = retinex_edgemask(aa_masked)
dn_merged = core.std.MaskedMerge(chromadn, merge1, linemask)

luma = get_y(dn_merged)
test1 = chroma_reconstruct(dn_merged, i444 = True)
test2 = dpir(test1, 10, cuda = 'trt', tiles = 2, fp16 = True)
test2 = core.std.ShufflePlanes([dn_merged, test2], [0, 1, 2], vs.YUV)

u_recon = plane(test2, 1)
v_recon = plane(test2, 2)

v_dh1 = fine_dehalo(v_recon, rx = 1.7, darkstr=0, brightstr=1)
v_double = v_recon.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose().znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose()
v_dh2 = fine_dehalo(v_double, rx = 2.0, darkstr=0, brightstr=1).resize.Point(1920, 1080)
v_dh = core.std.Expr([v_dh1, v_dh2], "x y min")

test3 = core.std.ShufflePlanes([dn_merged, u_recon, v_dh], [0, 0, 0], vs.YUV)
chromafix = core.resize.Bicubic(test3, format = vs.YUV420P16)


###DEBAND
deband = dbs.F3kdb.deband(chromafix, 17, [28, 36, 36], 0)
deband = adptvgrnMod(deband, strength = [0.04, 0], size = 1.2, sharp = 80, luma_scaling = 2, static = True)
deband = adptvgrnMod(deband, strength = [0.08, 0], size = 1.2, sharp = 80, luma_scaling = 8, static = False)

tmask = texture_mask(get_y(chromafix), 1)
debandmask = core.std.Expr([tmask, linemask], "x y max")
deband_masked = core.std.MaskedMerge(deband, chromafix, debandmask)


###OUTPUT
ed_splice = replace_ranges(deband_masked, ed, edranges)
grain = adptvgrnMod(deband_masked, strength = [0.02, 0], static = True, luma_scaling = 2, size = 1.2, sharp = 80)
grain = adptvgrnMod(grain, strength = [0.04, 0], static = False, luma_scaling = 6, size = 1.2, sharp = 80)
out = depth(grain, 10)


###WORKRAW
wr = dbs.F3kdb.deband(ed, 17, [48, 55, 55], 0)
wr = HQDeringmod(wr, mrad = 3, msmooth = 1, mthr = 20, nrmode = 1)
wr = fine_dehalo(wr, rx = 1.7, darkstr = 0, brightstr = 0.7)
wr = adptvgrnMod(wr, strength = [0.07, 0], static = True, luma_scaling = 2, size = 1.2, sharp = 80)
wr = depth(wr, 10)

out.set_output()





###OUTPUT
#borders.set_output(1)
#chromadn.set_output(2)
#test2.set_output(3)
#test3.set_output(4)
#chromafix.set_output(5)
