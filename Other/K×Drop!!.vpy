import vapoursynth as vs
from os import path
from functools import partial
import fvsfunc as fvf
import kagefunc as kgf
import mvsfunc as mvf
import vsutil
from vsutil import get_y, join, plane, get_w, depth
from adptvgrnMod import adptvgrnMod
import lvsfunc as lvf
import debandshit
import havsfunc as haf
import vardefunc as vdf
core = vs.core
core.max_cache_size = 1024 * 64


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
avc = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\AVC.mp4")
vp9 = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\VP9.webm")


###SPLICING THE SOURCES
src = vp9[0:303] + avc[303:435] + vp9[435:509] + avc[509:877] + vp9[877:921] + avc[921:1063] + vp9[1063:1615] + avc[1615:1899] + vp9[1899:1940] + avc[1940:2050] + vp9[2050:2094] + avc[2094:2284] + vp9[2284:2368] + avc[2368:2530] + vp9[2530:2750] + avc[2750:3103] + vp9[3103:3463] + avc[3463:3513] + vp9[3513:3875] + avc[3875:4115] + vp9[4115:4201] + avc[4201:4288] + vp9[4288:4301] + avc[4301:5088] + vp9[5088:5155] + avc[5155:5369] + vp9[5369:5448] + avc[5448:5524] + vp9[5524:5787] + avc[5787:6237] + vp9[6237:6319] + avc[6319:6464] + vp9[6464:6543] + avc[6543:7100] + vp9[7100:7336] + avc[7336:7494] + vp9[7494:8343]
src = fvf.Depth(src, 16)
deblock = fvf.AutoDeblock(src)


###DENOISE
ref = haf.SMDegrain(deblock, tr=3, RefineMotion=True, thSAD=120)
dn = mvf.BM3D(deblock, ref=ref, sigma=[1.5,0], profile1="lc")
dn = core.knlm.KNLMeansCL(dn, h=0.7, a=4, d=3, s=4, device_type='gpu', device_id=0, channels='UV')


###MASKS
luma_mask = lvf.denoise.adaptive_mask(src, luma_scaling=18)
line_mask = kgf.retinex_edgemask(dn, sigma=1)
aa_mask = core.std.Sobel(get_y(dn))
detail_mask = lvf.denoise.detail_mask(dn, brz_a=3000, brz_b=2000)

ed_gray = core.std.ShufflePlanes(dn, 0, vs.GRAY)
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


###DEBAND
dn = core.std.MaskedMerge(dn, deblock, line_mask)
weakdb = core.placebo.Deband(dn, planes = 1, iterations = 4, threshold = 4.0, radius = 18.0, grain = 4.0, dither = True, dither_algo = 0)
weakdb = core.placebo.Deband(weakdb, planes = 2|4, iterations = 2, threshold = 5.0, radius = 18.0, grain = 0.0, dither = True, dither_algo = 0)

strongdb = core.placebo.Deband(dn, planes = 1, iterations = 4, threshold = 5.5, radius = 18.0, grain = 5.0, dither = True, dither_algo = 0)
strongdb = core.placebo.Deband(strongdb, planes = 2|4, iterations = 2, threshold = 6.0, radius = 18.0, grain = 0.0, dither = True, dither_algo = 0)

db = core.std.MaskedMerge(weakdb, strongdb, luma_mask)
db = core.std.MaskedMerge(db, deblock, detail_mask)
db = core.std.MaskedMerge(db, deblock, texture_mask)

###AA
aa = lvf.sraa(db, rfactor=1.5)
aa = core.std.MaskedMerge(db, aa, aa_mask)

###OUTPUT
out = adptvgrnMod(aa, 0.2, 0.15, 1.25, luma_scaling=8, sharp=80, static=False, lo=19)
final = fvf.Depth(out, 10)
#comp = core.std.Interleave([src, out])
final.set_output()