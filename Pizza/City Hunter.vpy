import vapoursynth as vs
import fvsfunc as fvf
import kagefunc as kgf
from vsutil import get_y, join, plane
from adptvgrnMod import adptvgrnMod
import lvsfunc as lvf
import debandshit
import havsfunc as haf
core = vs.core
core.max_cache_size = 1024 * 48
core.num_threads = 6


#code by Varde, because f3kdb strength only changes every 16 steps    
def dumb3kdbv2(clip, radius=16, strength=24):
    div = (strength - 1) % 16
    if strength < 17:
        return clip
    if div == 0:
        return clip.f3kdb.Deband(radius, strength, strength, strength, 0, 0, output_depth=16)
    lo_str = strength - div
    hi_str = strength + 16 - div
    lo_clip = dumb3kdbv2(clip, radius, lo_str)
    hi_clip = dumb3kdbv2(clip, radius, hi_str)
    return core.std.Merge(lo_clip, hi_clip, (strength - lo_str) / 16)


###GETTING THE SOURCES
src = core.lsmas.LWLibavSource(r"E:\[BDMV][190130][City Hunter][BD-BOX]\Disc 7\BDMV\STREAM\00006.m2ts")
src = fvf.Depth(src, 16)
src = core.std.CropRel(src, 242, 242, 0, 0)

###DEBAND
weakdb = core.placebo.Deband(src, planes = 1, iterations = 4, threshold = 3.8, radius = 18.0, grain = 2.0, dither = True, dither_algo = 0)
weakdb = core.placebo.Deband(weakdb, planes = 2|4, iterations = 2, threshold = 4.0, radius = 18.0, grain = 0.0, dither = True, dither_algo = 0)

altdb = dumb3kdbv2(src)

luma_mask = lvf.denoise.adaptive_mask(src, luma_scaling=80)
line_mask = kgf.retinex_edgemask(src, sigma=1)
detail_mask = lvf.denoise.detail_mask(src, brz_a=4000, brz_b=2000)

db = core.std.MaskedMerge(altdb, weakdb, luma_mask)
db = core.std.MaskedMerge(db, src, line_mask)
db = core.std.MaskedMerge(db, src, detail_mask)


###OUTPUT
out = adptvgrnMod(db, strength=0.25, size=1.25, luma_scaling=30, sharp=80, static=False, lo=19)
final = fvf.Depth(out, 10)
#comp = core.std.Interleave([src, out])
final.set_output()