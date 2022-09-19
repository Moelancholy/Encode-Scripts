import vapoursynth as vs


core = vs.core


import debandshit as dbs
import vardefunc.mask as mask

from adptvgrnMod import adptvgrnMod
from kagefunc import retinex_edgemask
from havsfunc import SMDegrain, HQDeringmod
from muvsfunc import SSIM_downsample
from vardefunc.scale import nnedi3cl_double, to_444
from vsutil import get_depth, get_w, get_y, depth, iterate


###CUSTOM FUNCS
def merge_tvcr(tv, cr, radius=3, passes=2, lehmer=True):
    tv_high = core.std.MakeDiff(tv, tv.std.BoxBlur(hradius=radius, hpasses=passes, vradius=radius, vpasses=passes))
    tv_low = core.std.MakeDiff(tv, tv_high)
    cr_high = core.std.MakeDiff(cr, cr.std.BoxBlur(hradius=radius, hpasses=passes, vradius=radius, vpasses=passes))
    if lehmer:
        cr_hi = core.std.Expr([tv_high, cr_high], "x 32768 - 3 pow y 32768 - 3 pow + x 32768 - 2 pow y 32768 - 2 pow 0.0001 + + / 32768 +")
    return core.std.MergeDiff(tv_low, cr_hi)






###SOURCES
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 01 (Amazon dAnime CBR 1080p).mkv")
vbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Mobile Suit Gundam - The Witch from Mercury - 01 (Amazon dAnime VBR 1080p).mkv")

lehmer = merge_tvcr(vbr, cbr)
lehmer16 = depth(lehmer, 16)


###DERING
dering = HQDeringmod(lehmer16, mrad = 3, msmooth = 1, mthr = 20, nrmode = 1)


###DEBAND
deband = dbs.placebo_deband(dering, 18, 4, 3, 0)


###OUTPUT
grain = adptvgrnMod(deband, strength = 0.06, static = True, luma_scaling = 2, grain_chroma = False, size = 1.2, sharp = 75)
grain = adptvgrnMod(grain, strength = 0.11, static = False, luma_scaling = 10, grain_chroma = False, size = 1.2, sharp = 75)

out = depth(grain, 10)
out.set_output()
#out.set_output(2)
#deband2.set_output(3)
#deband3.set_output(4)

#10336