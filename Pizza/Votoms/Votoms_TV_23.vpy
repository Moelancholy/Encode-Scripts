import vapoursynth as vs

import G41Fun as gf
import havsfunc as haf
import lvsfunc as lvf
import vardefunc as vdf
import xvs

from rektlvl import rektlvls
from vsutil import *

core = vs.core
core.max_cache_size = 1024 * 16

src = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][210225] 装甲騎兵ボトムズ Blu-ray Perfect Soldier Box\VOTOMS_PERFECT_SOLDIER_BOX_D4\BDMV\STREAM\00004.m2ts")
crop = core.std.Crop(src, 240, 240, 0, 0)

fb = rektlvls(crop, rownum=[0,1, 1078, 1079], rowval=[0,1,1,1], colnum=[0,1,1438,1439], colval=[4,-4,-6,3])

y, u, v = split(depth(fb, 32))
dh = 540
dw = get_w(dh, 4 / 3)
descale_y =  depth(core.descale.Debilinear(y, dw, dh), 16)
downscale_y = depth(y.resize.Bicubic(dw, dh),16)
merge_y = core.std.Merge(descale_y, downscale_y, 0.6)
darken = haf.FastLineDarkenMOD(merge_y, strength=16)

dering = haf.EdgeCleaner(darken, 5, smode=1, hot=True, rmode=13)
dhmask = haf.HQDeringmod(dering, show=True)
dh = gf.MaskedDHA(dering, rx=1.6, ry=1.6, darkstr=0.00, brightstr=1.0, maskpull=25, maskpush=110)
dh = depth(core.std.MaskedMerge(dering, dh, dhmask), 32)

merge_u = u.resize.Bicubic(src_left=0.25)
merge_v = v.resize.Bicubic(src_left=0.25)
merge = join([dh, merge_u, merge_v])
cwarp = xvs.WarpFixChromaBlend(merge, thresh=100)
cwarpscale = cwarp.resize.Spline36(1440,1080)
cwarp10 = depth(cwarp, 10)

comp = core.std.Interleave([core.resize.Bicubic(depth(fb, 16), format = cwarpscale.format), cwarpscale])
cwarp10.set_output()