import vapoursynth as vs
from os import path
from functools import partial
import fvsfunc as fvf
import kagefunc as kgf
import mvsfunc as mvf
from vsutil import get_y, join, plane, get_w, depth
import vsutil
import xvs
import lvsfunc as lvf
import descale as dsc
import debandshit
from finedehalo import fine_dehalo
import havsfunc as haf
from vardefunc import diff_creditless_mask as DCM
import vardefunc as vdf
import bombzenfunc as bmf
core = vs.core
core.max_cache_size = 1024 * 48

###TEXTURE MASK
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
    

###GETTING THE SOURCES
src = core.ffms2.Source(r"C:\Users\twint\Desktop\Encode\[BDMV][140326] Anne of Green Gables\ANNE_OF_GREEN_GABLES_4\BDMV\STREAM\00005.m2ts")[:37001]
src = fvf.Depth(src, 16)
src = core.std.CropRel(src, 242, 242, 0, 0)


###PREP



###DENOISE
preden = core.knlm.KNLMeansCL(src, h=5.5, a=4, d=3, s=4, device_type='gpu', device_id=0, channels='Y')
detail_mask = lvf.denoise.detail_mask(preden, brz_a=3000, brz_b=5500)

ed_gray = core.std.ShufflePlanes(preden, 0, vs.GRAY)
rmask = debandshit.rangemask(ed_gray, 2, 2)
emask = ed_gray.std.Prewitt()

em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
em_hi = vsutil.iterate(em_hi, core.std.Minimum, 5)
em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
em_me = vsutil.iterate(em_me, core.std.Minimum, 4)
em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
em_lo = vsutil.iterate(em_lo, core.std.Minimum, 2)
rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')

weighted = LevelsM(rm_txt, points=[x *256 for x in (1.75, 2.5, 5, 10)], levels=[0, 1, 1, 0], xpass=[0,0], return_expr=0)
texture_mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')

ref = haf.SMDegrain(src, tr=3, RefineMotion=True, thSAD=420)
weakdenoise = mvf.BM3D(src, ref=ref, sigma=[4,0], profile1="vn", radius1=2)
strongdenoise = mvf.BM3D(src, ref=ref, sigma=[6,0], profile1="vn", radius1=2)
denoise = core.std.MaskedMerge(strongdenoise, weakdenoise, texture_mask)
denoise = core.knlm.KNLMeansCL(denoise, h=0.7, a=4, d=3, s=4, device_type='gpu', device_id=0, channels='UV')
denoise = haf.ContraSharpening(denoise, src)


###DEBAND
weakdb = core.placebo.Deband(denoise, planes = 1, iterations = 4, threshold = 3.0, radius = 18.0, grain = 6.0, dither = True, dither_algo = 0)
weakdb = core.placebo.Deband(weakdb, planes = 2|4, iterations = 2, threshold = 4.0, radius = 18.0, grain = 0.0, dither = True, dither_algo = 0)
weakdb = core.std.MaskedMerge(weakdb, src, detail_mask)
weakdb = core.std.MaskedMerge(weakdb, denoise, texture_mask)


###EDGES
ec = haf.EdgeCleaner(weakdb, strength=8, rmode=13, hot=True)
dehalo = fine_dehalo(ec, rx=2.4, ry=2.2, darkstr=0.1, brightstr=1)
dehalo = depth(dehalo, 32)

l_mask = lvf.util.quick_resample(dehalo, kgf.retinex_edgemask)
cwarp = core.std.MaskedMerge(xvs.WarpFixChromaBlend(dehalo, depth=16), dehalo, l_mask)
cwarp = core.rgsf.Repair(cwarp, dehalo, 13)
cwarp = fvf.Depth(cwarp, 16)


###OUTPUT
final = fvf.Depth(cwarp, 10)
#comp = core.std.Interleave([get_y(src), texture_mask])
final.set_output()