import vapoursynth as vs
core = vs.core

import havsfunc as haf
import vardefunc as vdf
import xvs

from kagefunc import crossfade
from vsutil import depth, get_y



###IMPORT SOURCES, FADE
ncop = core.lsmas.LWLibavSource(r"E:\Anime\Kimagure Orange Road\KIMAGURE ORANGE ROAD TV5\BDMV\STREAM\00013.m2ts")
crop = core.std.CropRel(ncop, 264, 216, 0, 0)
black = core.std.BlankClip(crop, length = 40)
fade = depth(crossfade(black, crop[5:], 35), 16)


###MASK, CONVERT RGBS
mask = vdf.mask.PrewittStd().get_mask(get_y(fade), multi=2.00).std.Minimum().std.Maximum().std.Maximum().std.Binarize(70<<7).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
fade444 = vdf.scale.to_444(depth(fade, 32), 1440, 1080, join_planes = True)
fade444 = core.resize.Bicubic(fade444, matrix_in = 1, primaries_in = 1, transfer_in = 1, matrix = 1, primaries = 1, transfer = 1)
fadeRGBS = core.resize.Bicubic(fade444, 1440, 1080, format = vs.RGBS)


###FIX CHROMA
cden = core.ccd.CCD(fadeRGBS, 20)
cden = core.std.ShufflePlanes([fade, core.resize.Bicubic(cden, format = vs.YUV420P16, matrix = 1)], [0, 1, 2], vs.YUV)
crest = xvs.mwcfix(cden, kernel = 1, restore = 5, a = 2, grad = 4, warp = 4, thresh = 128, blur = 3, repair = 1)
crest_m = core.std.MaskedMerge(cden, crest, mask)


###DERING
dering1 = haf.HQDeringmod(crest_m, mrad = 1, msmooth = 1, mthr = 60)
dering2 = haf.EdgeCleaner(dering1, strength = 5, rep = True, rmode = 17, smode = 1, hot = True)
dering_m = core.std.MaskedMerge(crest, dering2, mask)



###OUTPUT
out = depth(dering_m, 10)
out.set_output()
#dering3.set_output(6)